import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

def preprocess_image(image_path, img_size=224):

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0  # Normalize to [0,1]
    
    return img

def preprocess_mask(mask_path, img_size=224):

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")
        
    mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    mask = mask - 1  # Convert values to zero-based indexing
    
    # One-hot encode the mask
    mask_one_hot = to_categorical(mask, num_classes=3)
    
    return mask_one_hot
