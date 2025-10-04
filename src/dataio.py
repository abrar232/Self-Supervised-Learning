"""
Minimal image data I/O for self-supervised + simple supervised workflows (TensorFlow).

What this module does
---------------------
- Recursively **list images** under a root.
- Build an **unlabeled** `tf.data.Dataset` for self-supervised pretraining.
- Build a **labeled** dataset from a class-folder layout via Keras utility.

Design notes
------------
- Images are decoded to **RGB float32 in [0, 1]** and resized to `img_size × img_size`.
- We use `tf.data.AUTOTUNE` for map/prefetch.
- The decoder accepts **PNG/JPEG/BMP** (despite the function name, see docstring).
- If no images are found, an informative error is raised instead of silently continuing.

Quick usage
-----------
>>> ds = dataset_unlabeled("data/raw/images", img_size=224, batch_size=64)
>>> batch = next(iter(ds))
>>> batch.shape  # (64, 224, 224, 3)

>>> ds_sup = dataset_from_directory("data/raw", img_size=224, batch_size=64)
>>> images, labels = next(iter(ds_sup))
"""

from __future__ import annotations
from pathlib import Path
from typing import List
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


# -----------------------------------------------------------------------------
# File listing
# -----------------------------------------------------------------------------
def _is_image(fname: str) -> bool:
    """Return True if the filename looks like a supported image."""
    return fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))


def list_images(root: str | Path) -> list[Path]:
    """
    Recursively list image files under `root`.

    Parameters
    ----------
    root : str | Path
        Directory to search.

    Returns
    -------
    list[Path]
        Sorted paths to images.
    """
    root = Path(root)
    return sorted([p for p in root.rglob("*") if p.is_file() and _is_image(p.name)])


# -----------------------------------------------------------------------------
# Decoding / preprocessing
# -----------------------------------------------------------------------------
def decode_jpeg(path: tf.Tensor, img_size: int = 224) -> tf.Tensor:
    """
    Decode image file -> RGB float32 tensor resized to (img_size, img_size).

    Despite the name, this uses **tf.io.decode_image** under the hood so it
    supports **PNG, JPEG, BMP** uniformly.

    Parameters
    ----------
    path : tf.Tensor (dtype=string)
        Filesystem path to the image.
    img_size : int
        Output height/width.

    Returns
    -------
    tf.Tensor
        (H, W, 3) float32 in [0, 1]
    """
    bytes_ = tf.io.read_file(path)
    # decode_image supports multiple formats; channels=3 ensures RGB
    img = tf.io.decode_image(bytes_, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, (img_size, img_size))
    return img


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------
def dataset_unlabeled(
    root: str | Path,
    img_size: int = 224,
    batch_size: int = 64,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """
    Build an **unlabeled** dataset of images from any directory structure.

    Useful for self-supervised pretraining; labels are not used.

    Parameters
    ----------
    root : str | Path
        Directory containing images (any nested layout).
    img_size : int
        Resize target (square).
    batch_size : int
        Batch size.
    shuffle : bool
        Shuffle file list each epoch.

    Returns
    -------
    tf.data.Dataset
        Batches of images with shape (B, img_size, img_size, 3), float32 in [0,1].
    """
    paths = [str(p) for p in list_images(root)]
    if len(paths) == 0:
        raise FileNotFoundError(f"No images found under: {root}")

    ds = tf.data.Dataset.from_tensor_slices(paths)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)
    ds = ds.map(lambda p: decode_jpeg(p, img_size), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def dataset_from_directory(
    root: str | Path,
    img_size: int = 224,
    batch_size: int = 64,
) -> tf.data.Dataset:
    """
    Build a **labeled** dataset from a class-folder layout:

        root/
          class_a/*.jpg
          class_b/*.jpg
          ...

    The labels are integer-encoded in alphabetical order of subfolder names.

    Parameters
    ----------
    root : str | Path
        Root directory containing one subfolder per class.
    img_size : int
        Resize target (square).
    batch_size : int
        Batch size.

    Returns
    -------
    tf.data.Dataset
        Batches of (images, labels): (B, H, W, 3), (B,)
    """
    return (
        tf.keras.utils.image_dataset_from_directory(
            directory=root,
            image_size=(img_size, img_size),
            batch_size=batch_size,
            label_mode="int",
            shuffle=True,
        )
        .prefetch(AUTOTUNE)
    )
