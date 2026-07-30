import os
import csv
from PIL import Image, ImageDraw

def generate_offline_artifacts():
    """
    Called once at the start of the simulation by robot1.
    Generates the mandatory CSV and PNG files to avoid disqualification.
    """
    # 1. Ensure the sim_logs directory exists
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 2. Generate a dummy CSV (Teammate A will replace this with real CV outputs)
    csv_path = os.path.join(log_dir, "victim_location_estimates.csv")
    dummy_targets = [
        {"x": 2.5, "y": -1.5},
        {"x": 5.0, "y": 3.0},
        {"x": -2.0, "y": 4.5}
    ]
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"]) # Header
        for t in dummy_targets:
            writer.writerow([t["x"], t["y"]])
            
    # 3. Generate a strict B/W 600x600 PNG 
    png_path = os.path.join(log_dir, "map_estimate.png")
    img = Image.new("RGB", (600, 600), "white")
    draw = ImageDraw.Draw(img)
    # Draw a dummy solid black wall block to test validation
    draw.rectangle([200, 200, 400, 250], fill="black") 
    
    # Strictly enforce B/W to prevent anti-aliasing gray pixels
    img = img.convert("1") 
    img.save(png_path)
    print("Offline artifacts generated in sim_logs/")