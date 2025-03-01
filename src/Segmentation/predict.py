import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
from preprocess import preprocess_image
from train import iou_metric, dice_coefficient

def predict_segmentation(image_path, model_path="best_segmentation_checkpoint.h5", img_size=224):
    
    
    custom_objects = {
        'iou_metric': iou_metric,
        'dice_coefficient': dice_coefficient
    }
    model = load_model(model_path, custom_objects=custom_objects)
    
    # Preprocess the image 
    img = preprocess_image(image_path, img_size)
    
    # Make prediction
    prediction = model.predict(np.expand_dims(img, axis=0))
    mask = np.argmax(prediction[0], axis=-1)
    
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Realage")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="viridis")
    plt.title("Segmentation")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()
    
    return mask


from predict import predict_segmentation
mask = predict_segmentation("image.jpg", "best_segmentation_checkpoint.h5")
