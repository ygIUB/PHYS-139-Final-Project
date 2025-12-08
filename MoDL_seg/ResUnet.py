from tensorflow.keras.activations import relu
from tensorflow.keras.layers import (
    Input, Conv2D, Activation, MaxPooling2D,
    UpSampling2D, Concatenate, LayerNormalization
)
import tensorflow as tf


# ResUnet-Conv_block-DownSampling
def conv_block(input_x, kn1, kn2, kn3, side_kn):
    # Main pathway
    x = Conv2D(filters=kn1, kernel_size=(1, 1))(input_x)
    x = Activation(relu)(x)
    x = Conv2D(filters=kn2, kernel_size=(3, 3), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Activation(relu)(x)
    x = Conv2D(filters=kn3, kernel_size=(1, 1))(x)
    x = Activation(relu)(x)
    # Residual connection
    y = Conv2D(filters=side_kn, kernel_size=(1, 1))(input_x)
    y = MaxPooling2D(pool_size=(2, 2))(y)
    y = Activation(relu)(y)

    output = tf.keras.layers.add([x, y])
    output = Activation(relu)(output)
    return output


# ResUnet-Conv_block-UpSampling
def conv_block1(input_x, kn1, kn2, kn3, side_kn):
    # Main pathway
    x = Conv2D(filters=kn1, kernel_size=(1, 1))(input_x)
    x = Activation(relu)(x)
    x = Conv2D(filters=kn2, kernel_size=(3, 3), padding='same')(x)
    x = Activation(relu)(x)
    x = Conv2D(filters=kn3, kernel_size=(1, 1))(x)
    x = Activation(relu)(x)
    # Residual connection
    y = Conv2D(filters=side_kn, kernel_size=(1, 1))(input_x)
    y = Activation(relu)(y)

    output = tf.keras.layers.add([x, y])
    output = Activation(relu)(output)
    return output


# ResUnet-Identity_block
def identity_block(input_x, kn1, kn2, kn3):
    # Main pathway
    x = Conv2D(filters=kn1, kernel_size=(1, 1))(input_x)
    x = Activation(relu)(x)
    x = Conv2D(filters=kn2, kernel_size=(3, 3), padding='same')(x)
    x = Activation(relu)(x)
    x = Conv2D(filters=kn3, kernel_size=(1, 1))(x)
    x = Activation(relu)(x)

    output = tf.keras.layers.add([x, input_x])
    output = Activation(relu)(output)
    return output

def build_our_resunet(input_shape=(512, 512, 1)):
    """
    OurResUNet:
    - Encoder/decoder: ResUNet-style with residual blocks
    - Normalization: LayerNormalization (better for very small batch size)
    - Bottleneck: simple multi-scale context block (dilated conv with rates 1, 2, 4)
    - Output: 1-channel sigmoid mask
    """
    inputs = Input(shape=input_shape)

    # ---- Encoder ----
    # level 1
    c1 = Conv2D(64, (3, 3), padding='same')(inputs)
    c1 = LayerNormalization()(c1)
    c1 = Activation('relu')(c1)
    c1 = Conv2D(64, (3, 3), padding='same')(c1)
    c1 = LayerNormalization()(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling2D((2, 2))(c1)   # 256x256

    # level 2
    c2 = conv_block(p1, kn1=64, kn2=64, kn3=128, side_kn=128)  # 128 filters out + pooling inside block
    p2 = c2                                                   # already pooled to 128x128

    # level 3
    c3 = conv_block(p2, kn1=128, kn2=128, kn3=256, side_kn=256)  # 64x64
    p3 = c3

    # level 4
    c4 = conv_block(p3, kn1=256, kn2=256, kn3=512, side_kn=512)  # 32x32
    p4 = c4

    # ---- Bottleneck with multi-scale context ----
    b = Conv2D(512, (3, 3), padding='same')(p4)
    b = LayerNormalization()(b)
    b = Activation('relu')(b)

    # dilated conv branches
    b1 = Conv2D(512, (3, 3), padding='same', dilation_rate=1)(b)
    b1 = LayerNormalization()(b1)
    b1 = Activation('relu')(b1)

    b2 = Conv2D(512, (3, 3), padding='same', dilation_rate=2)(b)
    b2 = LayerNormalization()(b2)
    b2 = Activation('relu')(b2)

    b3 = Conv2D(512, (3, 3), padding='same', dilation_rate=4)(b)
    b3 = LayerNormalization()(b3)
    b3 = Activation('relu')(b3)

    b_cat = Concatenate(axis=3)([b1, b2, b3])
    b_out = Conv2D(512, (1, 1), padding='same')(b_cat)
    b_out = LayerNormalization()(b_out)
    b_out = Activation('relu')(b_out)   # 32x32, 512 channels

    # ---- Decoder ----
    # up 1: from 32x32 -> 64x64, match c3
    u6 = UpSampling2D(size=(2, 2))(b_out)           # 64x64
    u6 = Concatenate(axis=3)([u6, c3])
    u6 = conv_block1(u6, kn1=256, kn2=256, kn3=256, side_kn=256)

    # up 2: 64x64 -> 128x128, match c2
    u7 = UpSampling2D(size=(2, 2))(u6)              # 128x128
    u7 = Concatenate(axis=3)([u7, c2])
    u7 = conv_block1(u7, kn1=128, kn2=128, kn3=128, side_kn=128)

    # up 3: 128x128 -> 256x256, match c1
    u8 = UpSampling2D(size=(2, 2))(u7)              # 256x256
    u8 = Concatenate(axis=3)([u8, c1])
    u8 = Conv2D(64, (3, 3), padding='same')(u8)
    u8 = LayerNormalization()(u8)
    u8 = Activation('relu')(u8)
    u8 = Conv2D(64, (3, 3), padding='same')(u8)
    u8 = LayerNormalization()(u8)
    u8 = Activation('relu')(u8)

    # final up to 512x512 (optional, if you want symmetric structure)
    u9 = UpSampling2D(size=(2, 2))(u8)              # 512x512
    u9 = Conv2D(32, (3, 3), padding='same')(u9)
    u9 = LayerNormalization()(u9)
    u9 = Activation('relu')(u9)

    # ---- Output head ----
    outputs = Conv2D(1, (1, 1), activation='sigmoid', name='mask')(u9)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name="OurResUNet")
    return model
