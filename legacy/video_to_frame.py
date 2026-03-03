import cv2
import os

# Input path
video_path = "/data/lkolmar/datasets/emre_dataset2/data/00000/00000_frames.avi"

# Output folder
output_dir = "/data/lkolmar/datasets/emre_dataset2/data/00000/frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break  
    
    # Save frame as png
    frame_filename = os.path.join(output_dir, f"frame_{frame_count:05d}.png")
    cv2.imwrite(frame_filename, frame)
    
    frame_count += 1

cap.release()
print(f"Extracted: {frame_count} frames")