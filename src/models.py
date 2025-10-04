"""
Encoders and heads for self-supervised learning (TensorFlow/Keras).

What you get
------------
- `build_encoder(name="resnet50", img_size=224, trainable=True)` → Keras Model
    Backbone from `keras.applications` with **global average pooling** so it
    outputs a single feature vector per image.

- `build_projection_head(dim_in, dim_hidden=512, dim_out=128)` → Keras Model
    2-layer MLP that L2-normalizes outputs (common for SimCLR/InfoNCE).

- `build_linear_probe(dim_in, num_classes)` → Keras Model
    One-layer softmax classifier for **linear evaluation** on frozen features.

Notes
-----
- These encoders **do not apply `preprocess_input`** automatically. If you rely
  on ImageNet weights and want exact normalization, wrap inputs upstream with
  the appropriate `keras.applications.<Model>.preprocess_input` or use the
  same normalization you used during pretraining.
- To get `dim_in` for heads:
    >>> enc = build_encoder("resnet50", 224, trainable=False)
    >>> dim_in = enc.output_shape[-1]
- Switch backbones by name: `"resnet50"` (default) or `"efficientnetb0"`.
"""

from __future__ import annotations
import tensorflow as tf
from tensorflow import keras


def build_encoder(name: str = "resnet50", img_size: int = 224, trainable: bool = True) -> keras.Model:
    """
    Create a CNN backbone that outputs a **pooled feature vector**.

    Parameters
    ----------
    name : str, default="resnet50"
        Backbone identifier. Supported: "resnet50", "efficientnetb0".
    img_size : int, default=224
        Square input resolution (H=W=img_size).
    trainable : bool, default=True
        Whether to allow the backbone weights to update during training.

    Returns
    -------
    keras.Model
        A model mapping (B, H, W, 3) → (B, D), where D is the pooled feature dim.
    """
    # Define input tensor
    input_t = keras.Input((img_size, img_size, 3))

    # Choose a backbone with include_top=False and global average pooling
    if name.lower() == "efficientnetb0":
        base = keras.applications.EfficientNetB0(
            include_top=False, input_tensor=input_t, pooling="avg"
        )
    else:
        # Default to ResNet50
        base = keras.applications.ResNet50(
            include_top=False, input_tensor=input_t, pooling="avg"
        )

    # Freeze or unfreeze according to `trainable`
    base.trainable = trainable

    # Wrap in a clean-named model (inputs → pooled feature vector)
    return keras.Model(base.inputs, base.outputs, name=f"enc_{name}")


def build_projection_head(dim_in: int, dim_hidden: int = 512, dim_out: int = 128) -> keras.Model:
    """
    Build a **projection MLP** used in contrastive SSL (SimCLR-style).

    Parameters
    ----------
    dim_in : int
        Input feature dimension (e.g., encoder.output_shape[-1]).
    dim_hidden : int, default=512
        Hidden layer width.
    dim_out : int, default=128
        Output embedding dimension for the contrastive objective.

    Returns
    -------
    keras.Model
        Maps (B, dim_in) → (B, dim_out), **L2-normalized** along the last axis.
    """
    inp = keras.Input((dim_in,))

    # 2-layer MLP with ReLU, then L2-normalize embeddings
    x = keras.layers.Dense(dim_hidden, activation="relu")(inp)
    x = keras.layers.Dense(dim_out)(x)
    x = keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(x)

    return keras.Model(inp, x, name="proj_head")


def build_linear_probe(dim_in: int, num_classes: int) -> keras.Model:
    """
    Build a **linear classifier** (softmax) for evaluating frozen features.

    Parameters
    ----------
    dim_in : int
        Input feature dimension (encoder output size).
    num_classes : int
        Number of target classes.

    Returns
    -------
    keras.Model
        Maps (B, dim_in) → (B, num_classes) with softmax activation.
    """
    inp = keras.Input((dim_in,))
    out = keras.layers.Dense(num_classes, activation="softmax")(inp)
    return keras.Model(inp, out, name="linear_probe")
