import cv2
import os

def extract_frames(video_path, output_folder):
    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_folder, f'frame_{frame_number:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        frame_number += 1
    
    cap.release()
    print(f'Frames extracted to {output_folder}')

# Define paths
video_path = 'path_to_your_video.mp4'  # Replace with your video file path
output_folder = 'path_to_frames_folder'  # Replace with the desired output folder

# Extract frames
extract_frames(video_path, output_folder)
