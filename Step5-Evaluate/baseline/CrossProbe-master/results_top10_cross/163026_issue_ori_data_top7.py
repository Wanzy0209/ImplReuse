```python
import tensorflow as tf
import math

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> tf.Tensor:
    # Conversion: query.size(-2) -> tf.shape(query)[-2]
    L, S = tf.shape(query)[-2], tf.shape(key)[-2]
    
    # Conversion: math.sqrt(query.size(-1)) -> tf.math.sqrt(tf.cast(...))
    if scale is None:
        scale_factor = 1.0 / tf.math.sqrt(tf.cast(tf.shape(query)[-1], query.dtype))
    else:
        scale_factor = scale
        
    # Conversion: torch.zeros -> tf.zeros
    attn_bias = tf.zeros((L, S), dtype=query.dtype)
    
    if is_causal:
        assert attn_mask is None
        # Conversion: torch.ones(...).tril(...) -> tf.linalg.band_part
        # band_part(num_lower, num_upper) with -1, 0 creates a lower triangular matrix
        temp_mask = tf.linalg.band_part(tf.ones((L, S), dtype=tf.bool), -1, 0)
        
        # Conversion: masked_fill with -inf -> tf.where
        neg_inf = tf.cast(float("-inf"), query.dtype)
        attn_bias = tf.where(temp_mask, attn_bias, neg_inf)
        # Conversion: .to(query.dtype) -> tf.cast
        attn_bias = tf.cast(attn_bias, query.dtype)

    if attn_mask is not None:
        # Conversion: torch.bool -> tf.bool
        if attn_mask.dtype == tf.bool:
            neg_inf = tf.cast(float("-inf"), query.dtype)
            # Conversion: logical_not -> tf.logical_not (handled implicitly by tf.where order)
            attn_bias = tf.where(attn_mask, attn_bias, neg_inf)
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        # Conversion: repeat_interleave -> tf.repeat
        # Conversion: // -> tf.math.floordiv (or // for tensor scalars)
        repeats_k = tf.shape(query)[-3] // tf.shape(key)[-3]
        key = tf.repeat(key, repeats_k, axis=-3)
        
        repeats_v = tf.shape(query)[-3] // tf.shape(value)[-3]
        value = tf.repeat(value, repeats_v, axis=-3)

    # Conversion: query @ key.transpose(-2, -1) -> tf.matmul(..., transpose_b=True)
    attn_weight = tf.matmul(query, key, transpose_b=True) * scale_factor
    attn_weight = attn_weight + attn_bias
    # Conversion: torch.softmax -> tf.nn.softmax
    attn_weight = tf.nn.softmax(attn_weight, axis=-1)
    # Conversion: torch.dropout -> tf.nn.dropout
    attn_weight = tf.nn.dropout(attn_weight, rate=dropout_p)
    # Conversion: @ -> tf.matmul
    return tf.matmul(attn_weight, value)
```