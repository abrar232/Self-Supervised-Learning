from __future__ import annotations
import tensorflow as tf
from tensorflow import keras

def make_ssl_augmenter(img_size: int = 224) -> keras.Sequential:
    """
    Rough SimCLR-ish pipeline using Keras preprocessing layers.
    (Gaussian blur is omitted for portability.)
    """
    return keras.Sequential([
        keras.layers.RandomResizedCrop(img_size, img_size, scale=(0.2, 1.0)),
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.05),
        keras.layers.RandomZoom(0.2),
        keras.layers.RandomContrast(0.2),
        keras.layers.Rescaling(1.0/255.0),  # if inputs are uint8
        keras.layers.Normalization(mean=[0.485,0.456,0.406], variance=[0.229**2,0.224**2,0.225**2]),
    ], name="ssl_aug")

@tf.function
def pair_views(x, aug):
    """Return two independently augmented views of x."""
    return aug(x, training=True), aug(x, training=True)
