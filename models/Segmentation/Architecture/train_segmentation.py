import os
import cv2
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from segmentation_model import build_segmentation_model  

# Setting paths of the dataset
image_dir = r"D:\Oxford Data\data\oxford-iiit-pet\images"  
trimap_dir = r"D:\Oxford Data\data\oxford-iiit-pet\annotations\trimaps"  

# Making lists of all image and trimap files
image_files = sorted(glob(os.path.join(image_dir, "*.jpg")))
trimap_files = sorted(glob(os.path.join(trimap_dir, "*.png")))

# Processing the images
IMG_SIZE = 224

real_images = []  
segmented_images = [] 

num_samples = 1000  
sample_indices = np.random.choice(len(image_files), min(num_samples, len(image_files)), replace=False)

# Loop through selected sample indices and process images and trimaps
for idx in sample_indices:
    img_path = image_files[idx]
    trimap_path = trimap_files[idx]

    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Skipping unreadable image -> {img_path}")
        continue  # Skip unreadable images

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  
    img = img / 255.0  

    # Read and process the trimap (segmentation mask)
    trimap = cv2.imread(trimap_path, cv2.IMREAD_GRAYSCALE)
    if trimap is None:
        print(f"Warning: Skipping unreadable trimap -> {trimap_path}")
        continue  # Skip unreadable trimaps

    trimap = cv2.resize(trimap, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST) 
    trimap = trimap - 1  # Convert trimap to zero-based indexing

    # Append processed images and trimaps to the lists
    real_images.append(img)
    segmented_images.append(trimap)

# Convert lists to numpy arrays
real_images = np.array(real_images, dtype=np.float32)
segmented_images = np.array(segmented_images, dtype=np.int32)

print(f"Preprocessed {real_images.shape[0]} images and {segmented_images.shape[0]} trimaps.")

# One-hot encoding the segmentation labels
y_categorical = to_categorical(segmented_images, num_classes=3)

# Splitting off the test set (10% of the data)
x_temp, x_test, y_temp, y_test = train_test_split(real_images, y_categorical, test_size=0.1, random_state=42)

# Splitting the remaining data into training (70%) and validation (20%)
x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.22, random_state=42)

# Build the segmentation model using the pre-defined function in segmentation_model.py
model = build_segmentation_model(input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Callbacks
callbacks = [
    ModelCheckpoint('best_segmentation_model.h5', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, verbose=1)
]

# Training the model
BATCH_SIZE = 8
EPOCHS = 20

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)


# Save the trained model
model.save("models/final_segmentation_model.h5")
