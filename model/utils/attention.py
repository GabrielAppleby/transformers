import tensorflow as tf


def scaled_dot_product_attention(q, k, v):
    """
    Calculates the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """
    # matmul_qk = QK^T
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # Grab the depth
    # dk = tf.cast(tf.shape(k)[-1], tf.float32)

    # Scale matmul_qk (unscaled attention) by the sqrt of the depth
    # Apparently this helps with numerical stability
    # scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    # if mask is not None:
    #     scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1. (..., seq_len_q, seq_len_k)
    attention_weights = tf.nn.softmax(matmul_qk, axis=-1)

    # Attend to the values
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth)

    return output
