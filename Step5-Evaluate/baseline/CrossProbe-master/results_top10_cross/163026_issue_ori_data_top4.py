```python
import tensorflow as tf
import math

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> tf.Tensor:
    # Conversion: query.size(-2) -> tf.shape(query)[-2]
    L, S = tf.shape(query)[-2], tf.shape(key)[-2]
    
    # Conversion: math.sqrt(query.size(-1)) -> tf.sqrt(tf.cast(...))
    # We use tf.sqrt to handle dynamic tensor shapes correctly
    scale_factor = 1.0 / tf.sqrt(tf.cast(tf.shape(query)[-1], tf.float32)) if scale is None else scale
    
    # Conversion: torch.zeros(...) -> tf.zeros(...)
    attn_bias = tf.zeros((L, S), dtype=query.dtype)
    
    if is_causal:
        assert attn_mask is None
        # Conversion: torch.ones(...).tril(diagonal=0) -> tf.linalg.band_part(..., -1, 0)
        # band_part(num_lower, num_upper). -1 means full lower band, 0 means empty upper band.
        temp_mask = tf.linalg.band_part(tf.ones((L, S), dtype=tf.bool), -1, 0)
        
        # Conversion: masked_fill_(logical_not(), -inf) -> tf.where(mask, x, -inf)
        # We keep attn_bias where mask is True, otherwise -inf
        attn_bias = tf.where(temp_mask, attn_bias, tf.cast(float("-inf"), attn_bias.dtype))
        # Conversion: .to(query.dtype) is implicit in TF ops if types match

    if attn_mask is not None:
        # Conversion: dtype == torch.bool -> dtype == tf.bool
        if attn_mask.dtype == tf.bool:
            # Conversion: masked_fill_(logical_not(), -inf) -> tf.where(mask, x, -inf)
            attn_bias = tf.where(attn_mask, attn_bias, tf.cast(float("-inf"), attn_bias.dtype))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        # Conversion: repeat_interleave(..., dim=-3) -> tf.repeat(..., axis=-3)
        # Calculate repeats factor based on the number of heads
        repeats = tf.shape(query)[-3] // tf.shape(key)[-3]
        key = tf.repeat(key, repeats=repeats, axis=-3)
        value = tf.repeat(value, repeats=repeats, axis=-3)

    # Conversion: query @ key.transpose(-2, -1) -> tf.linalg.matmul(query, key, transpose_b=True)
    attn_weight = tf.linalg.matmul(query, key, transpose_b=True) * scale_factor
    attn_weight = attn_weight + attn_bias
    
    # Conversion: torch.softmax(..., dim=-1) -> tf.nn.softmax(..., axis=-1)
    attn_weight = tf.nn.softmax(attn_weight, axis=-1)
    
    # Conversion: torch.dropout(..., dropout_p, train=True) -> tf.nn.dropout(..., rate=dropout_p)
    # Note: tf.nn.dropout scales the output during training, similar to PyTorch's functional dropout
    if dropout_p > 0.0:
        attn_weight = tf.nn.dropout(attn_weight, rate=dropout_p)
        
    # Conversion: attn_weight @ value -> tf.linalg.matmul(attn_weight, value)
    return tf.linalg.matmul(attn_weight, value)
```