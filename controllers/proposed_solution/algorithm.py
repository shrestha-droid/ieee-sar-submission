import os
import csv
import cv2
import numpy as np

def generate_offline_artifacts():
    """
    Processes the drone video to generate map_estimate.png and victim_location_estimates.csv.
    """
    # Set up log directory right next to this script (creates controllers/proposed_solution/sim_logs)
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 1. Load the video using an absolute path to avoid terminal location issues
    video_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "recordings", "large_world_flyover.mp4"))
    cap = cv2.VideoCapture(video_path)
    
    # Grab a stable, high-altitude frame (e.g., frame 150 after takeoff)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 150)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"CRITICAL ERROR: Could not read video frame from {video_path}.")
        print("Did you install Git LFS, or is the path wrong?")
        return

    # 2. Extract the Map (Strict B/W)
    # Convert to grayscale and run Canny edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate and find contours to create SOLID walls, not just wireframes
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a pure white canvas matching the ORIGINAL FRAME SIZE first
    map_img = np.ones_like(gray) * 255
    
    # Draw solid black walls on the full-size canvas
    cv2.drawContours(map_img, contours, -1, (0), thickness=cv2.FILLED)
    
    # Now resize the finished map to the strict 600x600 canvas required by the rules
    resized_map = cv2.resize(map_img, (600, 600))
    
    # Force strict binary (no gray anti-aliasing allowed)
    _, binary_map = cv2.threshold(resized_map, 127, 255, cv2.THRESH_BINARY)
    
    # Save the PNG
    cv2.imwrite(os.path.join(log_dir, "map_estimate.png"), binary_map)

    # 3. Extract the Victims
    # Convert to HSV to find victim colors (teammate needs to tune these bounds!)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_color = np.array([0, 50, 50]) # Tune this
    upper_color = np.array([20, 255, 255]) # Tune this
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    victim_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Save the CSV
    csv_path = os.path.join(log_dir, "victim_location_estimates.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        
        for cnt in victim_contours:
            if cv2.contourArea(cnt) > 50: # Filter out tiny noise pixels
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # NOTE: Teammate must apply ArUco scale math here to convert cx/cy to real-world X/Y
                    world_x = cx * 0.05 # Placeholder math
                    world_y = cy * 0.05 # Placeholder math
                    
                    writer.writerow([world_x, world_y])
                    
    print(f"SUCCESS: Offline CV artifacts successfully generated in {log_dir}")

# --- STANDALONE TEST BLOCK ---
# This allows you to run `python map_maker.py` directly from PowerShell to test it!
if __name__ == "__main__":
    print("Executing Map Maker standalone test...")
    generate_offline_artifacts()