import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from Resnet50_model import build_resnet50

# Adding the path of the data for classification
base_path = r"D:\Oxford Data\data\oxford-iiit-pet"
images_dir = os.path.join(base_path, "images")
annotations_dir = os.path.join(base_path, "annotations")

# Reading the annotation files used for classification
trainval_file = os.path.join(annotations_dir, "trainval.txt")
test_file = os.path.join(annotations_dir, "test.txt")
list_file = os.path.join(annotations_dir, "list.txt")

# Function for reading annotation files
def read_annotation_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Remove any comment line in the file
    data_lines = [line.strip() for line in lines if not line.startswith('#')]
    
    # Making the data dataframe to make it easy to be understood
    data = []
    for line in data_lines:
        parts = line.split()
        if len(parts) >= 4:  
            image_name = parts[0]
            class_id = int(parts[1])
            species = int(parts[2])  
            breed_id = int(parts[3])
            data.append({
                'image_name': image_name,
                'class_id': class_id,
                'species': species,
                'breed_id': breed_id
            })
    
    return pd.DataFrame(data)

# Loading the annotations
trainval_df = pd.DataFrame(read_annotation_file(trainval_file))
test_df = pd.DataFrame(read_annotation_file(test_file))
list_df = pd.DataFrame(read_annotation_file(list_file))

num_of_test_samples = 1000

# Randomly selecting 1000 test samples without losing order
test_samples = test_df.sample(n=num_of_test_samples, random_state=42)

# Getting the remaining images
taken_samples = test_df.drop(test_samples.index)  

# Appending taken samples to trainval dataset without losing order
trainval_df = pd.concat([trainval_df, taken_samples], ignore_index=True)

# Making new test dataset
test_df = test_samples

# Function for preprocessing and attaching labels
def preprocessing_images(dataframe, images_dir, target_size=(224, 224), num_img=32):
    num_samples = len(dataframe)
    x = np.zeros((num_samples, target_size[0], target_size[1], 3), dtype=np.float32)
    y = np.zeros(num_samples, dtype=np.int32)
    
    for i, (_, row) in enumerate(dataframe.iterrows()):
        image_name = row['image_name']
        image_path = os.path.join(images_dir, f"{image_name}.jpg")
        
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            img = cv2.resize(img, target_size)
            img = preprocess_input(img)  # Preprocessing for ResNet50
            
            x[i] = img
            y[i] = row['class_id'] - 1  # Adjust class_id to zero-based index
            
        if (i + 1) % num_img == 0 or (i + 1) == num_samples:
            print(f"Processed {i + 1}/{num_samples} images", end='\r')
    
    print(f"\n{num_samples} images are completed")
    return x, y

# Preprocess training and testing data
x_train, y_train = preprocessing_images(trainval_df, images_dir)
x_test, y_test = preprocessing_images(test_df, images_dir)

# Function for splitting the data
def split_dataset(x, y, test_size=0.2, random_state=42):
    return train_test_split(x, y, test_size=test_size, random_state=random_state)

# Split data into training and validation sets
x_train, x_val, y_train, y_val = split_dataset(x_train, y_train)

# Load your pre-existing ResNet50 model from resnet50_model.py
model = build_resnet50(input_shape=(224, 224, 3), num_classes=37) 

# Data augmentation for training
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

# Callbacks
checkpoint = ModelCheckpoint(
    'best_checkpoints.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

callbacks = [checkpoint, early_stopping]

# Train the model
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    steps_per_epoch=len(x_train) // 32,
    epochs=15,
    validation_data=(x_val, y_val),
    callbacks=callbacks
)

# Save the trained model
model.save("models/final_resnet50_model.h5")

