```python
import tensorflow as tf
import math

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False):
    # Conversion: query.size(-2) -> tf.shape(query)[-2]
    L, S = tf.shape(query)[-2], tf.shape(key)[-2]
    
    # Conversion: Use tf.math.sqrt because query.size(-1) returns a Tensor in TF, not an int
    scale_factor = 1.0 / tf.math.sqrt(tf.cast(tf.shape(query)[-1], query.dtype)) if scale is None else scale
    
    # Conversion: torch.zeros -> tf.zeros
    attn_bias = tf.zeros((L, S), dtype=query.dtype)
    
    if is_causal:
        assert attn_mask is None
        # Conversion: torch.ones(...).tril -> tf.linalg.band_part
        # band_part(input, -1, 0) creates a lower triangular matrix
        temp_mask = tf.linalg.band_part(tf.ones((L, S), dtype=tf.bool), -1, 0)
        
        # Conversion: masked_fill_ -> tf.where
        # logical_not -> tf.logical_not
        attn_bias = tf.where(tf.logical_not(temp_mask), 
                             tf.cast(float("-inf"), query.dtype), 
                             attn_bias)
        # Conversion: .to(query.dtype) handled by tf.where cast argument

    if attn_mask is not None:
        # Conversion: torch.bool -> tf.bool
        if attn_mask.dtype == tf.bool:
            attn_bias = tf.where(tf.logical_not(attn_mask), 
                                 tf.cast(float("-inf"), query.dtype), 
                                 attn_bias)
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        # Conversion: repeat_interleave -> tf.repeat
        # Conversion: Integer division // -> tf.math.floordiv
        repeats = tf.math.floordiv(tf.shape(query)[-3], tf.shape(key)[-3])
        key = tf.repeat(key, repeats, axis=-3)
        value = tf.repeat(value, repeats, axis=-3)

    # Conversion: @ -> tf.matmul
    # Conversion: transpose(-2, -1) -> tf.linalg.matrix_transpose
    attn_weight = tf.matmul(query, tf.linalg.matrix_transpose(key)) * scale_factor
    attn_weight = attn_weight + attn_bias
    
    # Conversion: torch.softmax -> tf.nn.softmax
    attn_weight = tf.nn.softmax(attn_weight, axis=-1)
    
    # Conversion: torch.dropout(..., train=True) -> tf.nn.dropout
    # Note: TF functional dropout applies the mask immediately. 
    # The 'train=True' logic in PyTorch is implicit here as we are calling the op directly.
    attn_weight = tf.nn.dropout(attn_weight, rate=dropout_p)
    
    return tf.matmul(attn_weight, value)
```