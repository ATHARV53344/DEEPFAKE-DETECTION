import os
import shutil

def organize_images(source_dir, dest_real_dir, dest_fake_dir):
    # Iterate over the images in the source directory
    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                
                # Move images to appropriate folder
                if 'real' in folder_name.lower():
                    shutil.copy(img_path, dest_real_dir)
                elif 'fake' in folder_name.lower():
                    shutil.copy(img_path, dest_fake_dir)

# Define your source and destination directories
source_directory = 'C:/Users/91995/Downloads/archive/cropped_images'
real_images_directory = 'C:/Users/91995/Desktop/deepfake_detection/REAL'
fake_images_directory = 'C:/Users/91995/Desktop/deepfake_detection/FAKE'

# Ensure destination directories exist
os.makedirs(real_images_directory, exist_ok=True)
os.makedirs(fake_images_directory, exist_ok=True)

# Organize images
organize_images(source_directory, real_images_directory, fake_images_directory)

print("Images organized successfully.")
