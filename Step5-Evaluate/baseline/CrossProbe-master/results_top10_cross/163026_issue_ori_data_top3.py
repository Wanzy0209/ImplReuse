```python
import tensorflow as tf

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> tf.Tensor:
    # Get dynamic shapes for L and S (sequence lengths)
    L = tf.shape(query)[-2]
    S = tf.shape(key)[-2]
    
    # Calculate scale factor
    # Conversion: Use tf.sqrt and tf.cast to handle dynamic shapes and dtypes
    if scale is None:
        scale_factor = tf.cast(1.0 / tf.sqrt(tf.cast(tf.shape(query)[-1], tf.float32)), query.dtype)
    else:
        scale_factor = scale

    # Initialize attention bias
    # Conversion: tf.zeros creates a tensor filled with 0s
    attn_bias = tf.zeros((L, S), dtype=query.dtype)

    if is_causal:
        assert attn_mask is None
        # Conversion: tf.linalg.band_part creates a lower triangular matrix (tril equivalent)
        # -1 for num_lower means full lower triangle, 0 for num_upper means no upper triangle
        temp_mask = tf.linalg.band_part(tf.ones((L, S), dtype=tf.bool), -1, 0)
        # Conversion: tf.where replaces elements based on condition (masked_fill equivalent)
        attn_bias = tf.where(temp_mask, attn_bias, tf.cast(float("-inf"), attn_bias.dtype))
        # attn_bias.to(query.dtype) is implicit in the tf.where cast above

    if attn_mask is not None:
        if attn_mask.dtype == tf.bool:
            # Conversion: tf.where for boolean masking
            attn_bias = tf.where(attn_mask, attn_bias, tf.cast(float("-inf"), attn_bias.dtype))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        # Conversion: tf.repeat repeats elements along an axis (repeat_interleave equivalent)
        # Calculate repeats ratio for the head dimension (-3)
        repeats = tf.shape(query)[-3] // tf.shape(key)[-3]
        key = tf.repeat(key, repeats, axis=-3)
        value = tf.repeat(value, repeats, axis=-3)

    # Calculate attention scores
    # Conversion: tf.matmul with transpose_b=True handles query @ key.transpose(-2, -1)
    attn_weight = tf.matmul(query, key, transpose_b=True) * scale_factor
    
    # Add bias
    attn_weight = attn_weight + attn_bias
    
    # Softmax
    # Conversion: tf.nn.softmax applies softmax along the specified axis
    attn_weight = tf.nn.softmax(attn_weight, axis=-1)
    
    # Dropout
    # Conversion: tf.nn.dropout applies dropout. Note: TF scales output during training by default (1/(1-rate)),
    # which matches PyTorch's torch.dropout behavior when train=True.
    attn_weight = tf.nn.dropout(attn_weight, rate=dropout_p)
    
    return tf.matmul(attn_weight, value)
```