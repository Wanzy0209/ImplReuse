```python
import math

import tensorflow as tf


def manual_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
) -> tf.Tensor:
    """From https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html"""

    # L, S = query.size(-2), key.size(-2)
    L, S = tf.shape(query)[-2], tf.shape(key)[-2]
    
    # scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    # Using tf.math.sqrt and tf.cast to handle dynamic shapes and dtypes correctly
    if scale is None:
        scale_factor = 1.0 / tf.math.sqrt(tf.cast(tf.shape(query)[-1], query.dtype))
    else:
        scale_factor = scale

    # attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    # Device placement is handled by the context in TensorFlow
    attn_bias = tf.zeros((L, S), dtype=query.dtype)

    if is_causal:
        # assert attn_mask is None
        # temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        # attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        # attn_bias.to(query.dtype)
        # tf.linalg.band_part(input, -1, 0) creates a lower triangular matrix
        temp_mask = tf.linalg.band_part(tf.ones((L, S), dtype=tf.bool), -1, 0)
        attn_bias = tf.where(tf.logical_not(temp_mask), tf.cast(float("-inf"), query.dtype), attn_bias)

    if attn_mask is not None:
        # if attn_mask.dtype == torch.bool:
        #     attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        # else:
        #     attn_bias = attn_mask + attn_bias
        if attn_mask.dtype == tf.bool:
            attn_bias = tf.where(tf.logical_not(attn_mask), tf.cast(float("-inf"), query.dtype), attn_bias)
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        # key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        # value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)
        # PyTorch -3 corresponds to the 'heads' dimension (index 1 in B, H, S, D)
        repeats = tf.shape(query)[-3] // tf.shape(key)[-3]
        key = tf.repeat(key, repeats=repeats, axis=1)
        value = tf.repeat(value, repeats=repeats, axis=1)

    # attn_weight = query @ key.transpose(-2, -1) * scale_factor
    # tf.matmul with transpose_b=True handles key.transpose(-2, -1)
    attn_weight = tf.matmul(query, key, transpose_b=True) * scale_factor
    attn_weight = attn_weight + attn_bias
    
    # attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = tf.nn.softmax(attn_weight, axis=-1)
    
    # attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    # tf.nn.dropout rate is probability to drop (same as PyTorch dropout_p)
    if dropout_p > 0.0:
        attn_weight = tf.nn.dropout(attn_weight, rate=dropout_p)
        
    # return attn_weight @ value
    return tf.matmul(attn_weight, value)


batch_size, seq_len, num_heads, head_dim = 1, 8, 12, 64

# PyTorch: q = torch.randn(..., device="mps").transpose(1, 2)
# TensorFlow: Use /GPU:0 as the equivalent high-performance device.
# We create tensors in the GPU context.
with tf.device('/GPU:0'):
    # Shape (B, S, H, D)
    q = tf.random.normal((batch_size, seq_len, num_heads, head_dim))
    k = tf.random.normal((batch_size, seq_len, num_heads, head_dim))
    v = tf.random.normal((batch_size, seq_len, num_heads, head_dim))
    
    # Transpose to (B, H, S, D) equivalent to .transpose(1, 2)
    q = tf.transpose(q, perm=[0, 2, 1, 3])
    k = tf.transpose(k, perm=[0, 2, 1, 3])
    v = tf.transpose(v, perm=[0, 2, 1, 3])

# Create CPU copies equivalent to q.cpu()
with tf.device('/CPU:0'):
    q_cpu = tf.identity(q)
    k_cpu = tf.identity(k)
    v_cpu