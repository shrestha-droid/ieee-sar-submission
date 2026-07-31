from controller import Robot
import math
import os
import csv
from map_maker import generate_offline_artifacts
from radio_operator import build_victim_message, rapid_fire_burst

# Constants
MAX_SPEED = 6.0
WHEEL_RADIUS = 0.043
TRACK_WIDTH = 0.20  # Approximate distance between left and right wheels

# Handshake config: robot2 waits for this marker instead of assuming a
# fixed number of timesteps is enough for robot1 to finish writing the CSV.
READY_MARKER_NAME = ".map_ready"
MAX_HANDSHAKE_WAIT_STEPS = 200  # ~ a few seconds at typical timestep, tune as needed


class SARController:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.robot_id = self.robot.getName()

        # Init Motors
        self.motors = {}
        for n in ["fl_wheel_joint", "fr_wheel_joint", "rl_wheel_joint", "rr_wheel_joint"]:
            m = self.robot.getDevice(n)
            m.setPosition(float("inf"))
            m.setVelocity(0.0)
            self.motors[n] = m

        # Init Wheel Encoders for Odometry
        self.encoders = {}
        for n in ["front left wheel motor sensor", "front right wheel motor sensor",
                  "rear left wheel motor sensor", "rear right wheel motor sensor"]:
            e = self.robot.getDevice(n)
            e.enable(self.timestep)
            self.encoders[n] = e

        # Init Compass for absolute heading
        self.compass = self.robot.getDevice("imu compass")
        self.compass.enable(self.timestep)

        # Init Comms
        # NOTE: Webots' getDevice() typically returns None (with a printed
        # warning) for a missing device rather than raising -- don't rely on
        # the exception to catch a bad/missing device name.
        self.emitter = self.robot.getDevice("supervisor emitter")
        if self.emitter is None:
            print(f"[{self.robot_id}] WARNING: 'supervisor emitter' device not found; "
                  f"victim messages will not be sent.")

        # Odometry State
        self.x = -0.375  # Starting X offset per rules
        self.y = 0.375 if self.robot_id == "robot1" else 0.0  # Starting Y offset per rules
        self.last_encoder_val = 0.0

        # Target State
        self.targets = []
        self.current_target_idx = 0

        self.sim_logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_logs")
        self.ready_marker_path = os.path.join(self.sim_logs_dir, READY_MARKER_NAME)

        self.lidar = self.robot.getDevice("laser")
        if self.lidar:
            self.lidar.enable(self.timestep)
            self.lidar.enablePointCloud()
        else:
            print(f"[{self.robot_id}] WARNING: LiDAR device 'laser' not found.")

    def _wait_for_map_ready(self):
        """
        robot2 blocks here (stepping the sim so it doesn't desync) until
        robot1 signals the CSV is fully written, instead of trusting a
        single fixed-length step to be enough.
        """
        waited = 0
        while not os.path.exists(self.ready_marker_path):
            if self.robot.step(self.timestep) == -1:
                return
            waited += 1
            if waited > MAX_HANDSHAKE_WAIT_STEPS:
                print(f"[{self.robot_id}] WARNING: timed out waiting for map-ready marker; "
                      f"proceeding anyway, CSV may be incomplete.")
                return

    def load_and_split_targets(self):
        """Reads the CSV and splits it between robot1 and robot2."""
        csv_path = os.path.join(self.sim_logs_dir, "victim_location_estimates.csv")
        all_targets = []
        try:
            with open(csv_path, mode='r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    all_targets.append({"x": float(row[0]), "y": float(row[1])})
        except Exception as e:
            print(f"[{self.robot_id}] Could not load CSV: {e}")
            return

        # Coordination: Split the workload
        half = len(all_targets) // 2
        if self.robot_id == "robot1":
            self.targets = all_targets[:half]
        else:
            self.targets = all_targets[half:]

        # Visit nearest-first from our own start position instead of raw
        # CSV order, to cut down on zig-zagging.
        self.targets.sort(key=lambda t: (t["x"] - self.x) ** 2 + (t["y"] - self.y) ** 2)

        print(f"[{self.robot_id}] Assigned {len(self.targets)} targets.")

    def set_speed(self, left, right):
        self.motors["fl_wheel_joint"].setVelocity(left)
        self.motors["rl_wheel_joint"].setVelocity(left)
        self.motors["fr_wheel_joint"].setVelocity(right)
        self.motors["rr_wheel_joint"].setVelocity(right)

    def update_odometry(self):
        """Calculate X/Y position using encoders and compass."""
        # Calculate distance traveled by averaging the four encoders
        current_enc = sum(e.getValue() for e in self.encoders.values()) / 4.0
        if math.isnan(current_enc):
            return

        delta_enc = current_enc - self.last_encoder_val
        distance = delta_enc * WHEEL_RADIUS
        self.last_encoder_val = current_enc

        # Get absolute heading from compass
        comp_vals = self.compass.getValues()
        if not math.isnan(comp_vals[0]):
            heading = math.atan2(comp_vals[1], comp_vals[0])
            # Update coordinates
            self.x += distance * math.cos(heading)
            self.y += distance * math.sin(heading)

    def run(self):
        # Robot 1 acts as the team leader to generate the map first, then
        # signals readiness so robot2 doesn't read a half-written CSV.
        if self.robot_id == "robot1":
            os.makedirs(self.sim_logs_dir, exist_ok=True)
            #generate_offline_artifacts()
            with open(self.ready_marker_path, "w") as f:
                f.write("ready")
        else:
            self._wait_for_map_ready()

        # Give Webots a tick to establish files and sensors
        self.robot.step(self.timestep)
        self.load_and_split_targets()

        # THE MAIN CONTROL LOOP (This must wrap all movement logic)
        while self.robot.step(self.timestep) != -1:
            self.update_odometry()

            if self.current_target_idx >= len(self.targets):
                self.set_speed(0, 0)
                continue  # All assigned targets visited

            target = self.targets[self.current_target_idx]

            # Distance to target
            dx = target["x"] - self.x
            dy = target["y"] - self.y
            dist = math.sqrt(dx ** 2 + dy ** 2)

            # If we are within 1.0m, fire the message and move to next target
            if dist < 1.0:
                print(f"[{self.robot_id}] Reached target {self.current_target_idx}! Firing payload.")
                msg = build_victim_message(self.robot, self.x, self.y, confidence=0.85)
                rapid_fire_burst(self.emitter, msg)
                self.current_target_idx += 1
                continue

            # Simple proportional navigation
            comp_vals = self.compass.getValues()
            current_heading = math.atan2(comp_vals[1], comp_vals[0])
            target_heading = math.atan2(dy, dx)

            angle_error = target_heading - current_heading
            # Normalize angle to [-pi, pi]
            angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi

            turn_speed = max(-MAX_SPEED, min(MAX_SPEED, angle_error * 3.0))
            forward_speed = MAX_SPEED if abs(angle_error) < 0.5 else 0.0

            # --- LiDAR Reactive Avoidance ---
            avoidance_turn = 0.0
            if self.lidar:
                range_image = self.lidar.getRangeImage()
                if range_image:
                    # The RpLidarA2 returns an array. We want the forward-facing cone.
                    mid_idx = len(range_image) // 2
                    cone_size = int(len(range_image) * 0.1) # Check a 20% slice ahead
                    
                    front_left = range_image[mid_idx - cone_size : mid_idx]
                    front_right = range_image[mid_idx : mid_idx + cone_size]
                    
                    # Filter out 'inf' or 'nan' values safely
                    min_left = min([r for r in front_left if not math.isinf(r) and not math.isnan(r)] + [float('inf')])
                    min_right = min([r for r in front_right if not math.isinf(r) and not math.isnan(r)] + [float('inf')])
                    min_front = min(min_left, min_right)
                    
                    # If an object breaches the 0.5m threshold, hijack the steering
                    if min_front < 0.5:
                        forward_speed = MAX_SPEED * 0.4 # Slow down to avoid crashing
                        # Steer away from whichever side is closer to the wall
                        if min_left < min_right:
                            avoidance_turn = MAX_SPEED # Hard right
                        else:
                            avoidance_turn = -MAX_SPEED # Hard left

            # Apply final motor commands (Avoidance overrides waypoint steering if triggered)
            final_turn = avoidance_turn if avoidance_turn != 0.0 else turn_speed
            self.set_speed(forward_speed - final_turn, forward_speed + final_turn)

if __name__ == "__main__":
    controller = SARController()
    controller.run()
