import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

def load_and_preprocess_frames(directory):
    images = []
    labels = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            img_path = os.path.join(directory, filename)
            img = load_img(img_path, target_size=(128, 128))  # Resize frames to the size expected by your model
            img_array = img_to_array(img) / 255.0  # Normalize images
            images.append(img_array)
            labels.append(0)  # Dummy label for now
    
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def preprocess_frames_and_save():
    frames_folder = 'path_to_frames_folder'  # Replace with your frames folder path
    output_folder = 'path_to_preprocessed_frames'  # Replace with your desired output path
    
    X, y = load_and_preprocess_frames(frames_folder)

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Save the preprocessed frames
    np.save(os.path.join(output_folder, 'X.npy'), X)
    np.save(os.path.join(output_folder, 'y.npy'), y)

# Execute the preprocessing function
preprocess_frames_and_save()
