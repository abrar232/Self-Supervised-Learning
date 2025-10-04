"""
Self-supervised (SimCLR-ish) image augmentations with Keras preprocessing layers.

What this provides
------------------
- `make_ssl_augmenter(img_size=224)` → `keras.Sequential`
    A portable augmentation pipeline for contrastive learning:
    RandomResizedCrop → Flip → Rotation → Zoom → Contrast → Rescale → Normalize

- `pair_views(x, aug)` → (view1, view2)
    Apply the same augmenter **twice** to produce two independent views of `x`,
    as required by SimCLR/InfoNCE setups.

Notes
-----
- The Gaussian blur is omitted for portability; some TF environments lack a built-in
  blur layer. You can add a custom blur layer if desired.
- The `Normalization` layer is initialized with **ImageNet mean/variance** so
  it roughly matches common encoder expectations. If you adapt a different
  distribution, replace with your own normalization.
- Inputs are converted to float32 in **[0, 1]** by `Rescaling(1/255.)`.
"""

from __future__ import annotations
import tensorflow as tf
from tensorflow import keras


def make_ssl_augmenter(img_size: int = 224) -> keras.Sequential:
    """
    Build a SimCLR-style augmentation pipeline using Keras preprocessing layers.

    Parameters
    ----------
    img_size : int, default=224
        Side length for square crops/resizing.

    Returns
    -------
    keras.Sequential
        A callable augmenter that maps `(B, H, W, 3) -> (B, img_size, img_size, 3)`.

    Pipeline
    --------
    - RandomResizedCrop(img_size, img_size, scale=(0.2, 1.0))
    - RandomFlip("horizontal")
    - RandomRotation(±0.05 rad)
    - RandomZoom(0.2)
    - RandomContrast(0.2)
    - Rescaling(1/255.)       → uint8 → float32 in [0, 1]
    - Normalization(mean, var)  (ImageNet stats)
    """
    return keras.Sequential(
        [
            # Spatial / photometric augmentations
            keras.layers.RandomResizedCrop(img_size, img_size, scale=(0.2, 1.0)),
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.05),
            keras.layers.RandomZoom(0.2),
            keras.layers.RandomContrast(0.2),

            # Numeric normalization
            keras.layers.Rescaling(1.0 / 255.0),  # if inputs are uint8 [0..255]
            keras.layers.Normalization(
                mean=[0.485, 0.456, 0.406],
                variance=[0.229**2, 0.224**2, 0.225**2],
            ),
        ],
        name="ssl_aug",
    )


@tf.function
def pair_views(x: tf.Tensor, aug: keras.Sequential) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Produce two independent augmented views of the same batch.

    Parameters
    ----------
    x : tf.Tensor
        Input batch `(B, H, W, 3)`; dtype uint8 or float32.
    aug : keras.Sequential
        Augmenter returned by `make_ssl_augmenter`.

    Returns
    -------
    (tf.Tensor, tf.Tensor)
        Two tensors `(view1, view2)`, each `(B, img_size, img_size, 3)` float32.

    Notes
    -----
    - `aug(..., training=True)` is called to ensure stochastic layers sample
      new randomness on each call, yielding *different* views.
    """
    # Two *independent* passes through the augmenter → distinct stochastic draws
    return aug(x, training=True), aug(x, training=True)
