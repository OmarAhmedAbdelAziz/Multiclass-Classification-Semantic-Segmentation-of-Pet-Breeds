import os
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def create_model(num_classes=37):
    # Create base model with pre-trained ResNet50 weights
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, datagen, x_train, y_train, x_val, y_val, batch_size=32, epochs=15, checkpoint_path='best_checkpoints.h5'):
    # Callbacks
    checkpoint = ModelCheckpoint(
        checkpoint_path,
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

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(x_train) // batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks
    )
    
    return history, model

def run_training(base_path):
    from data_loader import load_pet_dataset
    from preprocess import preprocessing_images, split_dataset, create_data_generator
    
    # Load dataset
    trainval_df, test_df, list_df, images_dir = load_pet_dataset(base_path)
    
    # Preprocess images
    x_train_all, y_train_all = preprocessing_images(trainval_df, images_dir)
    
    # Split into train and validation
    x_train, x_val, y_train, y_val = split_dataset(x_train_all, y_train_all)
    
    # Create model
    model = create_model()
    
    # Create data generator
    datagen = create_data_generator()
    
    # Train model
    history, trained_model = train_model(model, datagen, x_train, y_train, x_val, y_val)
    
    return trained_model, x_val, y_val, list_df

