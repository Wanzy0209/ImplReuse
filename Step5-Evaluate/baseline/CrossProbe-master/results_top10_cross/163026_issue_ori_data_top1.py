```python
import tensorflow as tf
import math

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> tf.Tensor:
    # Conversion: Use tf.shape to get dynamic tensor dimensions
    L, S = tf.shape(query)[-2], tf.shape(key)[-2]
    
    # Conversion: Use tf.sqrt for tensor compatibility if scale is None
    if scale is None:
        scale_factor = 1.0 / tf.sqrt(tf.cast(tf.shape(query)[-1], query.dtype))
    else:
        scale_factor = scale

    # Conversion: tf.zeros creates a tensor with specified shape and dtype
    attn_bias = tf.zeros((L, S), dtype=query.dtype)
    
    if is_causal:
        assert attn_mask is None
        # Conversion: tf.linalg.band_part creates a lower triangular mask (tril)
        # -1 means keep lower triangle, 0 means keep 0 diagonals of upper triangle
        temp_mask = tf.linalg.band_part(tf.ones((L, S), dtype=tf.bool), -1, 0)
        
        # Conversion: tf.where replaces masked_fill
        # Note: Using -1e9 instead of -inf for numerical stability in TF softmax
        attn_bias = tf.where(temp_mask, attn_bias, tf.cast(-1e9, attn_bias.dtype))

    if attn_mask is not None:
        # Conversion: Check dtype using tf.bool
        if attn_mask.dtype == tf.bool:
            # Conversion: tf.where for boolean masking
            attn_bias = tf.where(attn_mask, attn_bias, tf.cast(-1e9, attn_bias.dtype))
        else:
            attn_bias = attn_bias + attn_mask

    if enable_gqa:
        # Conversion: tf.repeat replaces repeat_interleave
        # Note: query.size(-3) corresponds to axis -3
        key = tf.repeat(key, repeats=tf.shape(query)[-3] // tf.shape(key)[-3], axis=-3)
        value = tf.repeat(value, repeats=tf.shape(query)[-3] // tf.shape(value)[-3], axis=-3)

    # Conversion: tf.matmul with transpose_b=True replaces @ and .transpose(-2, -1)
    attn_weight = tf.matmul(query, key, transpose_b=True) * scale_factor
    attn_weight = attn_weight + attn_bias
    
    # Conversion: tf.nn.softmax replaces torch.softmax
    attn_weight = tf.nn.softmax(attn_weight, axis=-1)
    
    # Conversion: tf.nn.dropout replaces torch.dropout
    # Note: PyTorch's train=True is implicit in TF ops; rate matches dropout_p
    attn_weight = tf.nn.dropout(attn_weight, rate=dropout_p)
    
    return tf.matmul(attn_weight, value)
```