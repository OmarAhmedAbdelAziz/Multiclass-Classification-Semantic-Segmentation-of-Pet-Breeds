import os
import numpy as np
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img_processed = preprocess_input(img.astype(np.float32))
    
    return img, img_processed

def predict_breed(model, img_processed, breed_name_mapping):
    # Expand dimensions to create batch of size 1
    img_batch = np.expand_dims(img_processed, axis=0)
    
    # Make prediction
    prediction = model.predict(img_batch)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Map to breed name (add 1 to get back to original class_id)
    predicted_breed = breed_name_mapping[predicted_class + 1]

    return predicted_breed

def display_prediction(img, predicted_breed):

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_breed}")
    plt.axis('off')
    
    plt.show()