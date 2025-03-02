import os
import numpy as np
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocessing_images(dataframe, images_dir, target_size=(224, 224), num_img=32):

    num_samples = len(dataframe)
    x = np.zeros((num_samples, target_size[0], target_size[1], 3), dtype=np.float32)
    y = np.zeros(num_samples, dtype=np.int32)
    
    for i, (_, row) in enumerate(dataframe.iterrows()):
        image_name = row['image_name']
        image_path = os.path.join(images_dir, f"{image_name}.jpg")
        
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
            img = cv2.resize(img, target_size)
            img = preprocess_input(img)
            
            x[i] = img
            y[i] = row['class_id'] - 1  
        
        if (i + 1) % num_img == 0 or (i + 1) == num_samples:
            print(f"Processed {i + 1}/{num_samples} images", end='\r')
    
    print(f"\n{num_samples} images are completed")
    return x, y

def split_dataset(x, y, test_size=0.2, random_state=42):
    return train_test_split(x, y, test_size=test_size, random_state=random_state)

def create_data_generator():
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )