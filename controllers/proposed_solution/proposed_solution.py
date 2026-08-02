from controller import Robot
import json
import cv2
import numpy as np
import math
import heapq
import os
import random
import time
from steering_pid import PIDSteeringController

# =========================================================================
# 1. INITIALIZATION & UNIVERSAL MAP LOADING
# =========================================================================
rosbot = Robot()
timestep = int(rosbot.getBasicTimeStep())
robot_id = rosbot.getName()

random.seed(robot_id)

steering_engine = PIDSteeringController(kp=3.0, max_speed=6.0)

# --- System Constants ---
WAYPOINT_TOLERANCE = 0.20  
REROUTE_COOLDOWN = 3.0     
STALL_VELOCITY_THRESHOLD = 0.005 

class RobotState:
    EXPLORING = "EXPLORING"
    NAVIGATING = "NAVIGATING"
    RECOVERING = "RECOVERING"
    VERIFYING_VICTIM = "VERIFYING_VICTIM"

# --- Hardware Setup ---
motors = {
    "fl": rosbot.getDevice("fl_wheel_joint"),
    "fr": rosbot.getDevice("fr_wheel_joint"),
    "rl": rosbot.getDevice("rl_wheel_joint"),
    "rr": rosbot.getDevice("rr_wheel_joint")
}
for motor in motors.values():
    motor.setPosition(float('inf'))
    motor.setVelocity(0.0)

lidar = rosbot.getDevice("laser")
lidar.enable(timestep)
lidar.enablePointCloud()

camera = rosbot.getDevice("camera rgb")
camera.enable(timestep)

compass = rosbot.getDevice("imu compass")
compass.enable(timestep)

encoders = {
    "fl": rosbot.getDevice("front left wheel motor sensor"),
    "fr": rosbot.getDevice("front right wheel motor sensor"),
    "rl": rosbot.getDevice("rear left wheel motor sensor"),
    "rr": rosbot.getDevice("rear right wheel motor sensor")
}
for encoder in encoders.values():
    encoder.enable(timestep)

supervisor_emitter = rosbot.getDevice("supervisor emitter")
supervisor_emitter.setChannel(43)

def report_victim(estimated_x, estimated_y, confidence=0.95):
    message_dict = {
        "timestamp": rosbot.getTime(),
        "robot_id": robot_id,
        "position": [estimated_x, estimated_y, 0.0],
        "victim_found": True,
        "victim_confidence": confidence 
    }
    supervisor_emitter.send(json.dumps(message_dict).encode('utf-8'))
    print(f"[{robot_id}] VICTIM REPORTED AT ({estimated_x:.2f}, {estimated_y:.2f})")
    
    log_dir = 'sim_logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    csv_file = os.path.join(log_dir, 'victim_location_estimates.csv')
    
    if not os.path.isfile(csv_file):
        with open(csv_file, 'w') as f:
            f.write("Timestamp,Robot_ID,Estimated_X,Estimated_Y,Confidence\n")
            
    with open(csv_file, 'a') as f:
        f.write(f"{rosbot.getTime()},{robot_id},{estimated_x},{estimated_y},{confidence}\n")

# =========================================================================
# 3. ADVANCED COSTMAP & GRADIENT GENERATION
# =========================================================================
try:
    raw_map = cv2.imread('sim_logs/map_estimate.png', cv2.IMREAD_GRAYSCALE)
    if raw_map is None: raise FileNotFoundError
except:
    raw_map = np.ones((600, 600), dtype=np.uint8) * 255
    cv2.rectangle(raw_map, (30, 30), (570, 570), 0, 15)

ARENA_SIZE = 10.0 
GRID_SIZE = 60

# Downsample and threshold
small_map = cv2.resize(raw_map, (GRID_SIZE, GRID_SIZE))
_, binary_map = cv2.threshold(small_map, 127, 255, cv2.THRESH_BINARY)

# Distance Transform Costmap
dist_transform = cv2.distanceTransform(binary_map, cv2.DIST_L2, 5)
max_dist = np.max(dist_transform)
if max_dist > 0:
    cost_map = 1.0 - (dist_transform / max_dist)
else:
    cost_map = np.zeros_like(small_map, dtype=np.float32)

# Tuned safe collision mask
obstacle_mask = dist_transform < 2.0 

# Coverage Tracker
visited_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

current_x = -0.375
current_y = 0.375 if robot_id == "robot1" else 0.0
current_heading = 0.0
last_encoder_values = {"fl": 0.0, "fr": 0.0}
WHEEL_RADIUS = 0.0425

# =========================================================================
# 4. LOGIC ENGINES 
# =========================================================================

def update_odometry():
    global current_x, current_y, current_heading, last_encoder_values, visited_grid
    
    fl_val = encoders["fl"].getValue()
    fr_val = encoders["fr"].getValue()
    
    dist_left = (fl_val - last_encoder_values["fl"]) * WHEEL_RADIUS
    dist_right = (fr_val - last_encoder_values["fr"]) * WHEEL_RADIUS
    
    last_encoder_values["fl"] = fl_val
    last_encoder_values["fr"] = fr_val
    
    dist_center = (dist_left + dist_right) / 2.0
    compass_vals = compass.getValues()
    current_heading = math.atan2(compass_vals[1], compass_vals[0])
    
    current_x += dist_center * math.cos(current_heading)
    current_y += dist_center * math.sin(current_heading)
    
    # Mark current area as visited
    gx, gy = world_to_grid(current_x, current_y)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if 0 <= gx+dx < GRID_SIZE and 0 <= gy+dy < GRID_SIZE:
                visited_grid[gy+dy, gx+dx] = True

def detect_candidate():
    raw_img = camera.getImage()
    if not raw_img: return False
    img_array = np.frombuffer(raw_img, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    hsv_frame = cv2.cvtColor(img_array[:, :, :3], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, np.array([5, 120, 120]), np.array([25, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 1200: return True
    return False

def verify_depth():
    depth_data = lidar.getRangeImage()
    if depth_data and depth_data[len(depth_data) // 2] < 1.0:
        return True
    return False

def world_to_grid(wx, wy):
    offset = ARENA_SIZE / 2.0
    gx = int(np.clip((wx + offset) / ARENA_SIZE * GRID_SIZE, 0, GRID_SIZE - 1))
    gy = int(np.clip((offset - wy) / ARENA_SIZE * GRID_SIZE, 0, GRID_SIZE - 1))
    return (gx, gy)

def grid_to_world(gx, gy):
    offset = ARENA_SIZE / 2.0
    wx = (gx / float(GRID_SIZE)) * ARENA_SIZE - offset
    wy = offset - (gy / float(GRID_SIZE)) * ARENA_SIZE
    return (wx, wy)

def a_star_pathfind(start_world, target_world):
    start = world_to_grid(start_world[0], start_world[1])
    goal = world_to_grid(target_world[0], target_world[1])
    
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]

    path_grid = []
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            while current in came_from:
                path_grid.append(current)
                current = came_from[current]
            path_grid.reverse()
            break

        for dx, dy in neighbors:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if obstacle_mask[ny, nx]: continue
                
                # Tuned down gradient penalty to prevent A* from failing in doorways
                wall_penalty = cost_map[ny, nx] * 3.0 
                tentative_g = g_score[current] + math.hypot(dx, dy) + wall_penalty
                
                neighbor = (nx, ny)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = math.hypot(goal[0] - nx, goal[1] - ny)
                    heapq.heappush(open_set, (tentative_g + h, neighbor))

    if not path_grid:
        return [target_world]
        
    # Reverted path smoothing to raw grid nodes to stop corner cutting
    return [grid_to_world(x, y) for (x, y) in path_grid]

def generate_coverage_waypoints(num_points=6):
    global visited_grid
    safe_y, safe_x = np.where(~obstacle_mask)
    
    valid_waypoints = []
    for i in range(len(safe_x)):
        gx, gy = safe_x[i], safe_y[i]
        
        # Ignore if already visited
        if visited_grid[gy, gx]: continue
            
        wx, wy = grid_to_world(gx, gy)
        
        # Sector Splitting
        if "1" in robot_id and wx < 0: continue
        if "2" in robot_id and wx >= 0: continue
            
        valid_waypoints.append((wx, wy))
        
    if not valid_waypoints:
        print(f"[{robot_id}] Sector fully explored. Resetting coverage map.")
        visited_grid.fill(False) 
        return [(0.0, 0.0)]
        
    random.shuffle(valid_waypoints)
    return valid_waypoints[:num_points]

def stop_motors():
    for motor in motors.values():
        motor.setVelocity(0.0)

# =========================================================================
# 5. STATE MACHINE EXECUTION LOOP
# =========================================================================
print(f"[{robot_id}] Stabilized Navigation Engine Online.")
search_waypoints = generate_coverage_waypoints(num_points=4)
current_target = None
current_path = []

current_state = RobotState.EXPLORING
stuck_counter = 0
last_x, last_y = current_x, current_y
last_reroute_time = rosbot.getTime()

while rosbot.step(timestep) != -1:
    update_odometry()
    
    if current_state != RobotState.VERIFYING_VICTIM and detect_candidate():
        print(f"[{robot_id}] Visual candidate spotted. Verifying depth...")
        current_state = RobotState.VERIFYING_VICTIM

    if current_state == RobotState.VERIFYING_VICTIM:
        stop_motors()
        if verify_depth():
            report_victim(current_x, current_y)
            rosbot.step(1000) 
        current_state = RobotState.EXPLORING

    elif current_state == RobotState.EXPLORING:
        if not search_waypoints:
            search_waypoints = generate_coverage_waypoints(num_points=4)
        
        current_target = search_waypoints.pop(0)
        current_path = a_star_pathfind((current_x, current_y), current_target)
        
        if current_path:
            current_state = RobotState.NAVIGATING
            last_reroute_time = rosbot.getTime()
            
    elif current_state == RobotState.NAVIGATING:
        if math.hypot(current_target[0] - current_x, current_target[1] - current_y) < WAYPOINT_TOLERANCE:
            current_state = RobotState.EXPLORING
            continue

        distance_moved = math.hypot(current_x - last_x, current_y - last_y)
        if distance_moved < STALL_VELOCITY_THRESHOLD:
            stuck_counter += 1
        else:
            stuck_counter = 0

        last_x, last_y = current_x, current_y

        if stuck_counter > 40 and (rosbot.getTime() - last_reroute_time > REROUTE_COOLDOWN):
            current_state = RobotState.RECOVERING
            stuck_counter = 0
            continue

        # Removed the buggy LiDAR swerve to let the PID controller handle the raw A* path
        left_speed, right_speed = steering_engine.calculate_speeds(current_x, current_y, current_heading, current_path)

        motors["fl"].setVelocity(left_speed)
        motors["rl"].setVelocity(left_speed)
        motors["fr"].setVelocity(right_speed)
        motors["rr"].setVelocity(right_speed)

    elif current_state == RobotState.RECOVERING:
        print(f"[{robot_id}] STALLED. Executing recovery maneuver.")
        for motor in motors.values(): motor.setVelocity(-4.0)
        rosbot.step(600)
        
        motors["fl"].setVelocity(-5.0)
        motors["rl"].setVelocity(-5.0)
        motors["fr"].setVelocity(5.0)
        motors["rr"].setVelocity(5.0)
        rosbot.step(800)
        
        stop_motors()
        current_state = RobotState.EXPLORING