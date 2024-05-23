# image_classification.py

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import random
import string
import os
from werkzeug.utils import secure_filename
from datetime import datetime

# Define constants
IMG_HEIGHT = 180
IMG_WIDTH = 180
CLASS_NAMES = ['ba', 'ca', 'da', 'ga', 'ha', 'ja', 'ka', 'la', 'ma', 'na', 'nga', 'nya', 'pa', 'ra', 'sa', 'ta', 'wa', 'ya']
MODEL_PATH = '/Users/alamsyah/Documents/Apps/Machine Learning/aksara_sunda/ImageClassification.keras'

# Load model
model = load_model(MODEL_PATH)

def generate_random_string(length=12):
    # Define the characters to use in the random string
    characters = string.ascii_letters + string.digits
    # Generate the random string
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

def classify_image(image_path):
    #  Load image
    img = tf.keras.utils.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    
    # Image to array
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Make prediction
    predictions = model.predict(img_array)
    # Get score
    score = tf.nn.softmax(predictions[0])
    
    # Prepare the result
    result = {
        'img_path': image_path,
        'class': CLASS_NAMES[np.argmax(score)],
        'confidence': float(np.max(score))
    }
    
    return result

def save_uploaded_file(file):
    filename = secure_filename(file.filename)
    file_path = os.path.join('uploads', filename)
    file.save(file_path)
    return file_path

def save_training_data(image_path, label):
    # # Get current year and month
    # now = datetime.now()
    # year_month = now.strftime("%Y_%m_%d")
    
    # Create directory if it doesn't exist
    base_dir = f'datasets/train'
    
    # Create directory if it doesn't exist
    label_dir = os.path.join(base_dir, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    
    # Save the image in the correct label directory
    filename = os.path.basename(image_path)
    new_path = os.path.join(label_dir, filename)
    os.rename(image_path, new_path)
    return new_path

def retrain_model():
    # # Get current year and month
    # now = datetime.now()
    # year_month = now.strftime("%Y_%m_%d")
    
    # Create an image data generator for training
    train_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        f'datasets/train',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32,
        class_mode='sparse',
        shuffle=True
    )
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Retrain the model
    model.fit(train_generator, epochs=5)
    
    # Save the updated model
    model.save(MODEL_PATH)
    
    return "Model retrained successfully"
