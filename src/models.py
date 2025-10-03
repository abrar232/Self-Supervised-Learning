from __future__ import annotations
import tensorflow as tf
from tensorflow import keras

def build_encoder(name: str = "resnet50", img_size: int = 224, trainable: bool = True) -> keras.Model:
    """
    Backbone that outputs a pooled feature vector.
    Uses keras.applications to stay dependency-light.
    """
    input_t = keras.Input((img_size, img_size, 3))
    if name.lower() == "efficientnetb0":
        base = keras.applications.EfficientNetB0(include_top=False, input_tensor=input_t, pooling="avg")
    else:
        base = keras.applications.ResNet50(include_top=False, input_tensor=input_t, pooling="avg")
    base.trainable = trainable
    return keras.Model(base.inputs, base.outputs, name=f"enc_{name}")

def build_projection_head(dim_in: int, dim_hidden: int = 512, dim_out: int = 128) -> keras.Model:
    inp = keras.Input((dim_in,))
    x = keras.layers.Dense(dim_hidden, activation="relu")(inp)
    x = keras.layers.Dense(dim_out)(x)
    x = keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(x)  # normalize embeddings
    return keras.Model(inp, x, name="proj_head")

def build_linear_probe(dim_in: int, num_classes: int) -> keras.Model:
    inp = keras.Input((dim_in,))
    out = keras.layers.Dense(num_classes, activation="softmax")(inp)
    return keras.Model(inp, out, name="linear_probe")
