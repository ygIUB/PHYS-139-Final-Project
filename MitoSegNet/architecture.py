"""
MitoSegNet Architecture - Modified U-Net

Based on: "MitoSegNet: Easy-to-use Deep Learning Segmentation for Analyzing
Mitochondrial Morphology" (iScience, 2020)

Key modifications from standard U-Net:
- Removed Dropout layers
- Added BatchNormalization layers instead
- Improved validation loss and dice coefficient
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D,
    BatchNormalization, Concatenate, Conv2DTranspose
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def build_mitosegnet(input_shape=(512, 512, 1), learning_rate=1e-4):
    """
    Build MitoSegNet architecture (Modified U-Net)

    Args:
        input_shape: Input image shape (height, width, channels)
        learning_rate: Learning rate for Adam optimizer

    Returns:
        Compiled Keras model
    """
    inputs = Input(input_shape)

    # ========== ENCODER ==========
    # Block 1
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

    # Block 2
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    bn2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

    # Block 3
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    bn3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

    # Block 4
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    bn4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)

    # ========== BOTTLENECK ==========
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    bn5 = BatchNormalization()(conv5)

    # ========== DECODER ==========
    # Block 6
    up6 = Conv2DTranspose(512, 2, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(bn5)
    merge6 = Concatenate(axis=3)([bn4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    bn6 = BatchNormalization()(conv6)

    # Block 7
    up7 = Conv2DTranspose(256, 2, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(bn6)
    merge7 = Concatenate(axis=3)([bn3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    bn7 = BatchNormalization()(conv7)

    # Block 8
    up8 = Conv2DTranspose(128, 2, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(bn7)
    merge8 = Concatenate(axis=3)([bn2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    bn8 = BatchNormalization()(conv8)

    # Block 9
    up9 = Conv2DTranspose(64, 2, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(bn8)
    merge9 = Concatenate(axis=3)([bn1, up9])
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    bn9 = BatchNormalization()(conv9)

    # ========== OUTPUT ==========
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn9)
    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

    # Build and compile model
    model = Model(inputs=inputs, outputs=outputs, name='MitoSegNet')

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Dice coefficient metric for evaluation

    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Dice coefficient value
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


if __name__ == "__main__":
    # Test model creation
    print("Building MitoSegNet model...")
    model = build_mitosegnet()
    print("\nModel Summary:")
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")