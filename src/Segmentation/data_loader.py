import os
from glob import glob
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_pet_segmentation_dataset(image_dir, trimap_dir, img_size=224, num_samples=1000):
    
    # Making lists of all image and trimap files
    image_files = sorted(glob(os.path.join(image_dir, "*.jpg")))
    trimap_files = sorted(glob(os.path.join(trimap_dir, "*.png")))

    # Processing the images
    real_images = []  # Images
    segmented_images = []  # Trimaps (Actual Segmentation)

    if num_samples == -1 or num_samples > len(image_files):
        num_samples = len(image_files)
    
    # Sample random indices
    sample_indices = np.random.choice(len(image_files), num_samples, replace=False)

    for idx in sample_indices:
        img_path = image_files[idx]
        trimap_path = trimap_files[idx]

        # Read and check if the image is valid
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Skipping unreadable image -> {img_path}")
            continue  # Skip unreadable images

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        img = img / 255.0  

        # Read and check if the trimap is valid
        trimap = cv2.imread(trimap_path, cv2.IMREAD_GRAYSCALE)
        if trimap is None:
            print(f"Warning: Skipping unreadable trimap -> {trimap_path}")
            continue  # Skip unreadable trimaps

        trimap = cv2.resize(trimap, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        trimap = trimap - 1  # Convert trimap values to zero-based indexing 

        real_images.append(img)
        segmented_images.append(trimap)

    # Convert to numpy arrays
    real_images = np.array(real_images, dtype=np.float32)
    segmented_images = np.array(segmented_images, dtype=np.int32)

    print(f"Preprocessed {real_images.shape[0]} images and {segmented_images.shape[0]} trimaps.")

    # One-hot encoding the labels
    y_categorical = to_categorical(segmented_images, num_classes=3)

    # Split data into train, validation, and test sets
    x_temp, x_test, y_temp, y_test = train_test_split(real_images, y_categorical, test_size=0.1, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.22, random_state=42)
    
    return x_train, x_val, x_test, y_train, y_val, y_test