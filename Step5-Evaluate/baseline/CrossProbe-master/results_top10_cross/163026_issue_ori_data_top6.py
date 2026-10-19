```python
import tensorflow as tf

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> tf.Tensor:
    # Conversion: query.size(-2) -> tf.shape(query)[-2]
    L = tf.shape(query)[-2]
    S = tf.shape(key)[-2]
    
    # Conversion: math.sqrt(query.size(-1)) -> tf.sqrt(tf.cast(...))
    # Note: Cast to query.dtype to ensure type consistency during multiplication
    d_k = tf.cast(tf.shape(query)[-1], query.dtype)
    scale_factor = 1.0 / tf.sqrt(d_k) if scale is None else scale
    
    # Conversion: torch.zeros(...) -> tf.zeros(...)
    attn_bias = tf.zeros((L, S), dtype=query.dtype)
    
    if is_causal:
        # assert attn_mask is None
        # Conversion: torch.ones(...).tril(...) -> tf.linalg.band_part(...)
        # band_part(input, -1, 0) creates a lower triangular matrix
        temp_mask = tf.linalg.band_part(tf.ones((L, S), dtype=tf.bool), -1, 0)
        
        # Conversion: masked_fill_(..., float("-inf")) -> tf.where(..., ..., -inf)
        neg_inf = tf.cast(float("-inf"), query.dtype)
        attn_bias = tf.where(temp_mask, attn_bias, neg_inf)

    if attn_mask is not None:
        if attn_mask.dtype == tf.bool:
            # Conversion: masked_fill_(..., float("-inf")) -> tf.where(..., ..., -inf)
            neg_inf = tf.cast(float("-inf"), query.dtype)
            attn_bias = tf.where(attn_mask, attn_bias, neg_inf)
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        # Conversion: repeat_interleave(..., -3) -> tf.repeat(..., axis=-3)
        # Note: -3 corresponds to the 3rd dimension from the end
        repeats = tf.shape(query)[-3] // tf.shape(key)[-3]
        key = tf.repeat(key, repeats=repeats, axis=-3)
        value = tf.repeat(value, repeats=repeats, axis=-3)

    # Conversion: query @ key.transpose(-2, -1) -> tf.matmul(query, key, transpose_b=True)
    attn_weight = tf.matmul(query, key, transpose_b=True) * scale_factor
    
    attn_weight = attn_weight + attn_bias
    
    # Conversion: torch.softmax(..., dim=-1) -> tf.nn.softmax(..., axis=-1)
    attn_weight = tf.nn.softmax(attn_weight, axis=-1)
    
    # Conversion: torch.dropout(..., dropout_p, train=True) -> tf.nn.dropout(..., rate=dropout_p)
    # Note: tf.nn.dropout scales the output during training, similar to PyTorch's train=True
    attn_weight = tf.nn.dropout(attn_weight, rate=dropout_p)
    
    return tf.matmul(attn_weight, value)
```