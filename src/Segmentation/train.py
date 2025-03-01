import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from data_loader import load_pet_segmentation_dataset
    

def iou_metric(y_true, y_pred):

    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
    y_true = tf.cast(tf.argmax(y_true, axis=-1), tf.float32)

    # Calculate IoU for each class separately
    class_scores = []
    for i in range(3):  # 3 classes: background, foreground, outline
        true_class = tf.cast(tf.equal(y_true, i), tf.float32)
        pred_class = tf.cast(tf.equal(y_pred, i), tf.float32)

        intersection = tf.reduce_sum(true_class * pred_class)
        union = tf.reduce_sum(true_class) + tf.reduce_sum(pred_class) - intersection

        iou = (intersection + tf.keras.backend.epsilon()) / (union + tf.keras.backend.epsilon())
        class_scores.append(iou)

    # Return mean IoU across all classes
    return tf.reduce_mean(class_scores)

def dice_coefficient(y_true, y_pred):

    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
    y_true = tf.cast(tf.argmax(y_true, axis=-1), tf.float32)

    # Calculate Dice for each class
    class_scores = []
    for i in range(3):  # 3 classes
        true_class = tf.cast(tf.equal(y_true, i), tf.float32)
        pred_class = tf.cast(tf.equal(y_pred, i), tf.float32)

        intersection = tf.reduce_sum(true_class * pred_class)
        dice = (2. * intersection + tf.keras.backend.epsilon()) / (
            tf.reduce_sum(true_class) + tf.reduce_sum(pred_class) + tf.keras.backend.epsilon())
        class_scores.append(dice)

    # Return mean Dice across all classes
    return tf.reduce_mean(class_scores)

def create_segmentation_model(img_size=224, num_classes=3):

    input_shape = (img_size, img_size, 3)
    inputs = Input(input_shape)

    # Encoder (MobileNetV2)
    base_model = MobileNetV2(include_top=False, weights="imagenet", input_tensor=inputs)

    # Extract features from different layers for skip connections
    layer_names = [
        "block_1_expand_relu",   # 112x112
        "block_3_expand_relu",   # 56x56
        "block_6_expand_relu",   # 28x28
        "block_13_expand_relu",  # 14x14
        "block_16_project"       # 7x7
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    encoder = Model(inputs=inputs, outputs=layers)
    encoder.trainable = True

    # Decoder
    x = layers[-1]
    skips = layers[:-1]

    # Upsample and merge with skip connections
    for skip in reversed(skips):
        x = UpSampling2D(size=(2, 2))(x)
        concat = Concatenate()([x, skip])

        x = Conv2D(256, 3, padding="same")(concat)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(256, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

    # Final upsampling and output layer
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    outputs = Conv2D(num_classes, 1, activation="softmax")(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',  
        metrics=['accuracy', iou_metric, dice_coefficient]
    )
    
    return model

def train_segmentation_model(model, x_train, y_train, x_val, y_val, batch_size=8, epochs=20, checkpoint_path="best_segmentation_checkpoint.h5"):
    
    callbacks = [
        ModelCheckpoint(checkpoint_path, save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=10, verbose=1)
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return history

# Function to run the entire training pipeline
def run_training(image_dir, trimap_dir, img_size=224, batch_size=8, epochs=20):

    # Load dataset
    x_train, x_val, x_test, y_train, y_val, y_test = load_pet_segmentation_dataset(
        image_dir, trimap_dir, img_size
    )
    
    # Create model
    model = create_segmentation_model(img_size)
    
    # Train model
    history = train_segmentation_model(
        model, x_train, y_train, x_val, y_val, 
        batch_size=batch_size, epochs=epochs
    )
    
    return model, history, (x_train, y_train, x_val, y_val, x_test, y_test)