import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Concatenate, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam

def build_segmentation_model(input_shape=(224, 224, 3), num_classes=3):

    inputs = Input(input_shape)

    # Encoder (Feature Extractor)
    base_model = MobileNetV2(include_top=False, weights="imagenet", input_tensor=inputs)
    
    # Skip connection layers
    layer_names = [
        "block_1_expand_relu",   # 112x112
        "block_3_expand_relu",   # 56x56
        "block_6_expand_relu",   # 28x28
        "block_13_expand_relu",  # 14x14
        "block_16_project"       # 7x7
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]
    encoder = Model(inputs=inputs, outputs=layers)
    encoder.trainable = True

    # Decoder (Upsampling & Skip Connections)
    x = layers[-1]  
    skips = layers[:-1]

    for skip in reversed(skips):
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate()([x, skip])

        x = Conv2D(256, 3, padding="same")(x)
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

    return model
