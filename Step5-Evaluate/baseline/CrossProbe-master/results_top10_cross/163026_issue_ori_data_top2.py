```python
import tensorflow as tf
import math

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> tf.Tensor:
    # Conversion: Use tf.shape to handle dynamic tensor shapes in TensorFlow
    L, S = tf.shape(query)[-2], tf.shape(key)[-2]
    
    # Conversion: Use tf.sqrt and tf.cast for tensor operations to ensure compatibility with graph execution
    scale_factor = 1.0 / tf.sqrt(tf.cast(tf.shape(query)[-1], tf.float32)) if scale is None else scale
    
    # Conversion: tf.zeros creates a tensor on the current device by default
    attn_bias = tf.zeros((L, S), dtype=query.dtype)
    
    if is_causal:
        assert attn_mask is None
        # Conversion: tf.linalg.band_part creates the lower triangular mask (1s in lower, 0s in upper)
        # We invert this (1 - band_part) to identify the upper triangle which needs -inf
        temp_mask = 1 - tf.linalg.band_part(tf.ones((L, S)), -1, 0)
        # Conversion: tf.where replaces masked_fill_ for conditional assignment
        attn_bias = tf.where(temp_mask > 0, tf.cast(float("-inf"), query.dtype), attn_bias)

    if attn_mask is not None:
        # Conversion: Compare with tf.bool instead of torch.bool
        if attn_mask.dtype == tf.bool:
            # Conversion: tf.logical_not and tf.where for boolean masking
            attn_bias = tf.where(tf.logical_not(attn_mask), tf.cast(float("-inf"), query.dtype), attn_bias)
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        # Conversion: tf.repeat replaces repeat_interleave
        # Calculate the repetition factor based on the head dimension (-3)
        repeats = tf.shape(query)[-3] // tf.shape(key)[-3]
        key = tf.repeat(key, repeats=repeats, axis=-3)
        value = tf.repeat(value, repeats=repeats, axis=-3)

    # Conversion: tf.matmul with transpose_b=True replaces @ and .transpose(-2, -1)
    attn_weight = tf.matmul(query, key, transpose_b=True) * scale_factor
    attn_weight = attn_weight + attn_bias
    
    # Conversion: tf.nn.softmax replaces torch.softmax
    attn_weight = tf.nn.softmax(attn_weight, axis=-1)
    
    # Conversion: tf.nn.dropout replaces torch.dropout
    # Note: In TF functional API, dropout is applied during training by default if rate > 0.
    # The 'train=True' argument in PyTorch ensures activation, which is the default behavior here.
    attn_weight = tf.nn.dropout(attn_weight, rate=dropout_p)
    
    return tf.matmul(attn_weight, value)
```