"""
ourResUnet.py

This file contains a first skeleton of our custom ResUNet-style architecture ("OurResUNet").

Design goals:
- A U-Net–like encoder–decoder structure with skip connections.
- Using simple residual blocks (ResNet idea) inside each stage.
- Later we can extend this with:
  * LayerNorm instead of BatchNorm for small batch sizes.
  * Multi-scale context at the bottleneck (dilated convolutions).
  * Optional boundary head or attention modules.

Right now this is intentionally minimal and easy to modify.
"""

from tensorflow.keras.layers import (
    Input, Conv2D, Activation, MaxPooling2D,
    UpSampling2D, Concatenate, LayerNormalization
)
from tensorflow.keras.models import Model


def residual_block(x, filters, name=None):
    """
    Basic residual block:

    - Two 3x3 convolutions with LayerNormalization + ReLU.
    - Optional 1x1 projection on the shortcut if the number of channels changes.

    Later we can:
    - Swap LayerNormalization for GroupNorm / BatchNorm if needed.
    - We can insert a lightweight attention module (e.g. SE / CBAM) inside this block.
    """
    shortcut = x

    y = Conv2D(filters, (3, 3), padding="same",
               name=None if name is None else name + "_conv1")(x)
    y = LayerNormalization(name=None if name is None else name + "_ln1")(y)
    y = Activation("relu", name=None if name is None else name + "_relu1")(y)

    y = Conv2D(filters, (3, 3), padding="same",
               name=None if name is None else name + "_conv2")(y)
    y = LayerNormalization(name=None if name is None else name + "_ln2")(y)

    # If the number of channels changes, project the shortcut to the same shape.
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), padding="same",
                          name=None if name is None else name + "_proj")(shortcut)

    out = Activation("relu", name=None if name is None else name + "_out")(y + shortcut)
    return out


def build_our_resunet(input_shape=(512, 512, 1)):
    """
    Skeleton for OurResUNet.

    Encoder path:
        - Several downsampling stages with residual blocks (residual_block)
    Bottleneck:
        - Currently a single residual block.
        - Later we will plug in a multi-scale context module
          (e.g., dilated convolutions with rates {1, 2, 4}).
    Decoder path:
        - Upsampling with skip connections, again using residual blocks.
    Output:
        - 1-channel sigmoid mask for binary segmentation.

    """
    inputs = Input(shape=input_shape, name="input")

    # ---- Encoder ----
    # Level 1: 512x512 -> 256x256
    e1 = residual_block(inputs, 64, name="enc1")
    p1 = MaxPooling2D(pool_size=(2, 2), name="pool1")(e1)   # 256x256

    # Level 2: 256x256 -> 128x128
    e2 = residual_block(p1, 128, name="enc2")
    p2 = MaxPooling2D(pool_size=(2, 2), name="pool2")(e2)   # 128x128

    # Level 3: 128x128 -> 64x64
    e3 = residual_block(p2, 256, name="enc3")
    p3 = MaxPooling2D(pool_size=(2, 2), name="pool3")(e3)   # 64x64

    # Level 4: 64x64 -> 32x32
    e4 = residual_block(p3, 512, name="enc4")
    p4 = MaxPooling2D(pool_size=(2, 2), name="pool4")(e4)   # 32x32

    # ---- Bottleneck (placeholder) ----
    # For now we just use one residual block.
    # Later we can replace this with a multi-scale dilated context module.
    b = residual_block(p4, 512, name="bottleneck")          # 32x32

    # ---- Decoder ----
    # Up 1: 32x32 -> 64x64
    u1 = UpSampling2D(size=(2, 2), name="up1")(b)
    u1 = Concatenate(axis=-1, name="concat1")([u1, e4])
    d1 = residual_block(u1, 512, name="dec1")               # 64x64

    # Up 2: 64x64 -> 128x128
    u2 = UpSampling2D(size=(2, 2), name="up2")(d1)
    u2 = Concatenate(axis=-1, name="concat2")([u2, e3])
    d2 = residual_block(u2, 256, name="dec2")               # 128x128

    # Up 3: 128x128 -> 256x256
    u3 = UpSampling2D(size=(2, 2), name="up3")(d2)
    u3 = Concatenate(axis=-1, name="concat3")([u3, e2])
    d3 = residual_block(u3, 128, name="dec3")               # 256x256

    # Up 4: 256x256 -> 512x512
    u4 = UpSampling2D(size=(2, 2), name="up4")(d3)
    u4 = Concatenate(axis=-1, name="concat4")([u4, e1])
    d4 = residual_block(u4, 64, name="dec4")                # 512x512

    # ---- Output head ----
    outputs = Conv2D(1, (1, 1), activation="sigmoid", name="mask")(d4)

    model = Model(inputs=inputs, outputs=outputs, name="OurResUNet")
    return model
