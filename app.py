from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
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
    img = load_img(img_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_video(video_path, model_path, output_folder):
    extract_frames(video_path, output_folder)
    model = load_model(model_path)
    predictions = []
    for filename in os.listdir(output_folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(output_folder, filename)
            frame = preprocess_frame(img_path)
            prediction = model.predict(frame)
            predictions.append(prediction)
    average_prediction = np.mean(predictions)
    result = 'Fake' if average_prediction > 0.5 else 'Real'
    return result

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)
            
            # Define paths
            model_path = 'C:/Users/91995/Desktop/deepfake_detection/deepfake_detection_model.h5'
            output_folder = 'extracted_frames'
            
            # Run prediction
            result = predict_video(video_path, model_path, output_folder)
            
            return render_template('result.html', result=result)
    return render_template('upload.html')

if __name__ == '__main__':
    # Update host to '0.0.0.0' to allow access from other devices
    app.run(host='0.0.0.0', port=5000, debug=True)
