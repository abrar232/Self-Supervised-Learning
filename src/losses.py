"""
Contrastive losses for self-supervised learning (TensorFlow).

This module provides the **NT-Xent (InfoNCE)** loss used in SimCLR-style
self-supervised learning. It operates on a batch of *paired* embeddings
(two augmented "views" for each image).

Assumptions
-----------
- `z1` and `z2` are **L2-normalized** along the feature dimension so that
  their dot product equals cosine similarity.
- Each row i in `z1` corresponds to row i in `z2` (positive pairs).

References
----------
- SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
  (Chen et al., 2020)
"""

from __future__ import annotations
import tensorflow as tf


def ntxent_loss(z1: tf.Tensor, z2: tf.Tensor, temperature: float = 0.1) -> tf.Tensor:
    """
    SimCLR **NT-Xent (InfoNCE)** loss for a batch of paired embeddings.

    Parameters
    ----------
    z1 : tf.Tensor, shape [batch, dim]
        Embeddings from view A. **Assumed L2-normalized** across `dim`.
    z2 : tf.Tensor, shape [batch, dim]
        Embeddings from view B. **Assumed L2-normalized** across `dim`.
    temperature : float, default=0.1
        Softmax temperature; smaller values sharpen the distribution.

    Returns
    -------
    tf.Tensor, scalar
        Mean NT-Xent loss over the 2B logits rows.

    Notes
    -----
    - We build a 2B×2B similarity matrix:
        z = concat([z1, z2])          → [2B, D]
        sim[i, j] = dot(z[i], z[j])   → cosine similarity if inputs are normalized
      Then we **mask out** self-similarities and apply a cross-entropy objective
      where each sample's positive is its augmented counterpart
      (i ↔ i+B and i+B ↔ i).
    """
    # Batch size (dynamic)
    batch = tf.shape(z1)[0]

    # Stack views to form a single matrix of size [2B, D]
    z = tf.concat([z1, z2], axis=0)  # [2B, D]

    # Pairwise similarity (cosine if z is L2-normalized)
    sim = tf.matmul(z, z, transpose_b=True) / temperature  # [2B, 2B]

    # Mask the diagonal (self-similarity) so it never becomes the target
    # Create a mask with zeros on the diagonal and ones elsewhere
    mask = tf.ones_like(sim) - tf.eye(2 * batch)
    # Replace diagonal with a very large negative number so softmax ~ 0 there
    sim = sim * mask + (-1e9) * (1 - mask)

    # Build target indices of positives:
    # for rows 0..B-1 → positives are B..2B-1 (i → i+B)
    # for rows B..2B-1 → positives are 0..B-1 (i → i-B)
    labels = tf.concat([tf.range(batch, 2 * batch), tf.range(0, batch)], axis=0)

    # Cross-entropy over rows of sim; labels pick the positive column per row
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=sim)

    # Return mean loss across the 2B rows
    return tf.reduce_mean(loss)
