import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

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

def preprocess_frame(img_path):
    img = load_img(img_path, target_size=(128, 128))  # Resize to match model input
    img_array = img_to_array(img) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

def predict_video(video_path, model_path, output_folder):
    # Extract frames from video
    extract_frames(video_path, output_folder)
    
    # Load the trained model
    model = load_model(model_path)
    
    # Predict each frame
    predictions = []
    for filename in os.listdir(output_folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(output_folder, filename)
            frame = preprocess_frame(img_path)
            prediction = model.predict(frame)
            predictions.append(prediction)
    
    # Process predictions (e.g., average prediction for video)
    average_prediction = np.mean(predictions)
    
    # Determine result
    result = 'Fake' if average_prediction > 0.5 else 'Real'
    print(f'The video is predicted to be: {result}')

# Define paths
video_path = 'path_to_your_video.mp4'  # Replace with the path to your video
model_path = 'C:/Users/91995/Desktop/deepfake_detection/deepfake_detection_model.h5'  # Replace with the path to your trained model
output_folder = 'path_to_extracted_frames'  # Replace with the folder where frames will be saved

# Run prediction
predict_video(video_path, model_path, output_folder)
