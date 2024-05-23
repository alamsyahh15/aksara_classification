# app.py

from flask import Flask, request, jsonify
from image_classification import classify_image, save_uploaded_file, generate_random_string, save_training_data, retrain_model
import os
import tensorflow as tf

app = Flask(__name__)

# Ensure the upload folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/classify', methods=['POST'])
def classify():
    if 'upload_file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['upload_file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file:
        file_path = save_uploaded_file(file)
        result = classify_image(file_path)
        return jsonify(result), 200

    return jsonify({"error": "Unknown error occurred"}), 500

@app.route('/classify-url', methods=['POST'])
def classify_url():
    data = request.get_json()
    if 'image_url' not in data:
        return jsonify({"error": "No image URL provided"}), 400

    image_url = data['image_url']
    random_string = generate_random_string()
    image_path = tf.keras.utils.get_file('image_upload_' + random_string + ".jpg", origin=image_url)
    result = classify_image(image_path)
    return jsonify(result), 200

@app.route('/retrain', methods=['POST'])
def retrain():
    data = request.get_json()
    if 'image_path' not in data or 'correct_label' not in data:
        return jsonify({"error": "Image path and correct label must be provided"}), 400

    image_path = data['image_path']
    correct_label = data['correct_label']

    # Save the image to the correct training directory
    save_training_data(image_path, correct_label)

    # Retrain the model
    retrain_message =  retrain_model()

    return jsonify({"message": retrain_message}), 200

if __name__ == '__main__':
    app.run(debug=True)
