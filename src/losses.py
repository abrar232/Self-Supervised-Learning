from __future__ import annotations
import tensorflow as tf

def ntxent_loss(z1: tf.Tensor, z2: tf.Tensor, temperature: float = 0.1) -> tf.Tensor:
    """
    SimCLR NT-Xent loss for a batch of paired embeddings.
    z1, z2: [batch, dim], assumed L2-normalized.
    """
    batch = tf.shape(z1)[0]
    z = tf.concat([z1, z2], axis=0)                           # [2B, D]
    sim = tf.matmul(z, z, transpose_b=True) / temperature     # [2B, 2B]
    # mask self similarity
    mask = tf.ones_like(sim) - tf.eye(2*batch)
    sim = sim * mask + (-1e9) * (1 - mask)

    # positives: (i, i+B) and (i+B, i)
    labels = tf.concat([tf.range(batch, 2*batch), tf.range(0, batch)], axis=0)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=sim)
    return tf.reduce_mean(loss)
