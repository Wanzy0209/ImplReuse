```python
import tensorflow as tf
import math

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> tf.Tensor:
    # Conversion: PyTorch size() to TensorFlow shape
    L, S = tf.shape(query)[-2], tf.shape(key)[-2]
    
    # Conversion: Use tf.sqrt for tensor compatibility instead of math.sqrt
    if scale is None:
        scale_factor = 1.0 / tf.sqrt(tf.cast(tf.shape(query)[-1], tf.float32))
    else:
        scale_factor = scale
        
    # Conversion: torch.zeros to tf.zeros
    attn_bias = tf.zeros((L, S), dtype=query.dtype)
    
    if is_causal:
        assert attn_mask is None
        # Conversion: torch.ones(...).tril to tf.linalg.band_part
        # band_part(num_lower, num_upper). -1 means infinite lower band.
        temp_mask = tf.linalg.band_part(tf.ones((L, S), dtype=tf.bool), -1, 0)
        # Conversion: masked_fill to tf.where
        # Fill with -inf where mask is False (upper triangle)
        attn_bias = tf.where(temp_mask, attn_bias, tf.cast(float("-inf"), attn_bias.dtype))
        attn_bias = tf.cast(attn_bias, query.dtype)

    if attn_mask is not None:
        # Conversion: torch.bool to tf.bool
        if attn_mask.dtype == tf.bool:
            # Conversion: masked_fill to tf.where
            attn_bias = tf.where(attn_mask, attn_bias, tf.cast(float("-inf"), attn_bias.dtype))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        # Conversion: repeat_interleave to tf.repeat
        # Calculate repeats based on the -3 dimension (usually heads)
        repeats = tf.shape(query)[-3] // tf.shape(key)[-3]
        key = tf.repeat(key, repeats=repeats, axis=-3)
        value = tf.repeat(value, repeats=repeats, axis=-3)

    # Conversion: @ operator with transpose to tf.matmul with transpose_b=True
    attn_weight = tf.matmul(query, key, transpose_b=True) * scale_factor
    attn_weight = attn_weight + attn_bias
    
    # Conversion: torch.softmax to tf.nn.softmax
    attn_weight = tf.nn.softmax(attn_weight, axis=-1)
    
    # Conversion: torch.dropout to tf.nn.dropout
    # Note: TF dropout rate matches PyTorch dropout_p (probability of dropping)
    attn_weight = tf.nn.dropout(attn_weight, rate=dropout_p)
    
    # Conversion: @ operator to tf.matmul
    return tf.matmul(attn_weight, value)
```