from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

def _is_image(fname: str) -> bool:
    return fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))

def list_images(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted([p for p in root.rglob("*") if p.is_file() and _is_image(p.name)])

def decode_jpeg(path: tf.Tensor, img_size: int = 224) -> tf.Tensor:
    bytes_ = tf.io.read_file(path)
    img = tf.io.decode_jpeg(bytes_, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (img_size, img_size))
    return img

def dataset_unlabeled(
    root: str | Path, img_size: int = 224, batch_size: int = 64, shuffle: bool = True
) -> tf.data.Dataset:
    """Images from any directory structure, unlabeled."""
    paths = [str(p) for p in list_images(root)]
    ds = tf.data.Dataset.from_tensor_slices(paths)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)
    ds = ds.map(lambda p: decode_jpeg(p, img_size), num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)

def dataset_from_directory(
    root: str | Path, img_size: int = 224, batch_size: int = 64
) -> tf.data.Dataset:
    """
    Labeled directory layout:
    root/
      class_a/*.jpg
      class_b/*.jpg
    """
    return tf.keras.utils.image_dataset_from_directory(
        directory=root,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="int",
        shuffle=True
    ).prefetch(AUTOTUNE)
