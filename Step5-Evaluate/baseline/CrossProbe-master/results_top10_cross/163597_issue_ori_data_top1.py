```python
import math

import tensorflow as tf


def manual_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
) -> tf.Tensor:
    """From https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html"""

    # Get dynamic shapes for L and S
    L = tf.shape(query)[-2]
    S = tf.shape(key)[-2]
    
    # Calculate scale factor
    scale_factor = 1.0 / math.sqrt(float(query.shape[-1])) if scale is None else scale
    
    # Initialize attention bias
    attn_bias = tf.zeros((L, S), dtype=query.dtype)
    
    if is_causal:
        assert attn_mask is None
        # Create a lower triangular mask (1s on and below diagonal)
        # PyTorch: torch.ones(L, S).tril(diagonal=0)
        # TensorFlow: tf.linalg.band_part keeps the lower triangle
        ones = tf.ones((L, S), dtype=tf.bool)
        lower_tri = tf.linalg.band_part(ones, -1, 0)
        # We need the upper triangle to be -inf
        # PyTorch: masked_fill(temp_mask.logical_not(), -inf)
        # TensorFlow: tf.where(~lower_tri, -inf, attn_bias)
        attn_bias = tf.where(
            tf.logical_not(lower_tri), 
            tf.cast(-float('inf'), query.dtype), 
            attn_bias
        )

    if attn_mask is not None:
        if attn_mask.dtype == tf.bool:
            # PyTorch: masked_fill(attn_mask.logical_not(), -inf)
            # TensorFlow: tf.where(~attn_mask, -inf, attn_bias)
            attn_bias = tf.where(
                tf.logical_not(attn_mask), 
                tf.cast(-float('inf'), query.dtype), 
                attn_bias
            )
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        # PyTorch: repeat_interleave(dim=-3)
        # TensorFlow: tf.repeat(axis=-3)
        # query.size(-3) // key.size(-3)
        repeats = query.shape[-3] // key.shape[-3]
        key = tf.repeat(key, repeats=repeats, axis=-3)
        value = tf.repeat(value, repeats=repeats, axis=-3)

    # Matmul: query @ key.transpose(-2, -1)
    # TensorFlow: tf.matmul(query, key, transpose_b=True)
    attn_weight = tf.matmul(query, key, transpose_b=True) * scale_factor
    attn_weight = attn_weight + attn_bias
    
    # Softmax
    attn_weight = tf.nn.softmax(attn_weight, axis=-1)
    
    # Dropout
    # PyTorch: torch.dropout(attn_weight, dropout_p, train=True)
    # TensorFlow: tf.nn.dropout(attn_weight, rate=dropout_p)
    # Note: TF dropout is scaled by default (inverted dropout), matching PyTorch's train=True
    if dropout_p > 0.0:
        attn_weight = tf.nn.dropout(attn_weight, rate=dropout_p)
        
    return tf.matmul(attn_weight, value)


batch_size, seq_len, num_heads, head_dim = 1, 8, 12, 64

# Use GPU context to simulate PyTorch's "mps" device
with tf.device('/GPU:0'):
    # PyTorch: randn(B, L, H, D).transpose(1, 2) -> (B, H, L, D)
    q_raw = tf.random.normal((batch_size, seq_len, num_heads, head_dim))
    k_raw = tf.random.normal((batch_size, seq_len, num_heads, head_dim))
    v_raw = tf.random.normal((batch_size, seq_len, num_heads, head_dim))

    q = tf.transpose(q_raw, perm=[0, 2, 1, 3])
    k = tf.transpose(k_raw, perm=[0, 2, 1, 3])
    v = tf.transpose(v_raw, perm=[0, 2, 1, 3])

    # Built-in
    # PyTorch: F.scaled_dot_product_attention
    # TensorFlow: tf.nn.scaled_dot_product_attention (Available in TF 2.16+)
    out_mps_builtin = tf.nn.scaled_dot_product_attention(q, k, v)
    # TF tensors are immutable, "contiguous" concept doesn't apply directly to memory layout in the same way
    out_mps_cont_builtin = tf.nn.scaled_dot_product_attention(q, k, v)

    # Manual
    out_mps_manual = manual_scaled_dot_product_attention(q, k, v)
    out_mps_cont_manual = manual_scaled_dot_product_attention(q, k, v)

# CPU versions
with tf.device('/CPU:0'):
    # Copy tensors to CPU context
    q_cpu = tf.identity(q)
    k_cpu = tf.identity(k)
    v_cpu = tf.identity(v)

    out_cpu_builtin = tf.nn.s