"""
OurResUNet – how this structure relates to the original ResUNet

High-level idea
---------------
We want a model that is clearly U-Net/ResUNet-style, but written in our own
words so that (1) we understand every piece, and (2) we have room to add our
own ideas (normalization, multi-scale, etc.).

Mapping to the original ResUNet blocks
--------------------------------------
In the reference ResUNet implementation there are three main types of blocks:

- conv_block
    * encoder/downsampling residual block
    * does: residual convs + downsampling (pooling) inside the block

- conv_block1
    * decoder/upsampling residual block
    * does: upsampling + skip concatenation + residual convs

- identity_block
    * residual block that keeps the same spatial resolution
    * does: F(x) + x at fixed HxW

In this file we keep the **same roles**, but we make them more explicit:

1) residual_block(...)
   --------------------
   This is our generic residual unit. It corresponds to the "inner part" of
   conv_block / conv_block1 / identity_block:

       y = Conv -> Norm -> ReLU -> Conv -> Norm
       out = ReLU(y + shortcut)

   We use LayerNormalization here (instead of BatchNorm) because our training
   batch size on GPU is very small (1–2), and BatchNorm becomes unstable in
   that regime. The shortcut uses a 1x1 conv if the number of channels changes.

2) Encoder stages (e1–e4 + p1–p4)
   -------------------------------
   Each encoder level is:

       eX = residual_block(...)
       pX = MaxPooling2D(eX)

   This plays the same role as a conv_block in the original ResUNet:
   "do residual convs at the current resolution, then downsample".
   We just split it into two clear steps instead of hiding pooling inside the block.

3) Bottleneck (bottleneck residual_block)
   --------------------------------------
   At the lowest resolution (32x32 in our current setting) we apply one
   residual_block. This is equivalent to an identity_block at the bottleneck
   in ResUNet.

   The current version is deliberately minimal. The plan is to replace or
   extend this with a multi-scale context module (e.g. several dilated
   convolutions with different rates) so that the network can better capture
   elongated mitochondria and multi-scale EM structures.

4) Decoder stages (u1–u4 / d1–d4)
   -------------------------------
   Each decoder level is:

       uX = UpSampling2D(...)
       uX = Concatenate([uX, encoder_feature])
       dX = residual_block(uX, ...)

   This corresponds to conv_block1 in the original ResUNet:
   "upsample, concatenate the skip connection, then apply a residual block".

Summary
-------
- We **keep the overall U-Net / ResUNet structure**:
  encoder–decoder with skips and residual connections.
- We **rewrite the blocks in a simpler, more explicit way** so that we can:
  * reason about each step (downsample, upsample, skip, residual),
  * swap BatchNorm -> LayerNorm for small-batch training,
  * later plug in multi-scale / attention / boundary heads.

This is why the architecture is structurally similar to ResUNet, but the code
is our own implementation, and it is designed to be extended in the next steps.
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
    - Insert a lightweight attention module (e.g. SE / CBAM) inside this block.
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
