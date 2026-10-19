```python
import tensorflow as tf
import math

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> tf.Tensor:
    # Conversion: PyTorch .size() -> TF .shape
    L, S = query.shape[-2], key.shape[-2]
    
    # Conversion: math.sqrt works with scalar values
    scale_factor = 1 / math.sqrt(query.shape[-1]) if scale is None else scale
    
    # Conversion: torch.zeros -> tf.zeros
    attn_bias = tf.zeros((L, S), dtype=query.dtype)
    
    if is_causal:
        assert attn_mask is None
        # Conversion: torch.ones(...).tril(...) -> tf.linalg.band_part(tf.ones(...), -1, 0)
        # Creates a lower triangular boolean mask
        temp_mask = tf.linalg.band_part(tf.ones((L, S), dtype=tf.bool), -1, 0)
        # Conversion: masked_fill_ -> tf.where
        # Note: logical_not is handled by inverting the condition in tf.where
        attn_bias = tf.where(temp_mask, attn_bias, float("-inf"))
        attn_bias = tf.cast(attn_bias, query.dtype)

    if attn_mask is not None:
        # Conversion: torch.bool -> tf.bool
        if attn_mask.dtype == tf.bool:
            # Conversion: masked_fill_ -> tf.where
            attn_bias = tf.where(attn_mask, attn_bias, float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        # Conversion: repeat_interleave -> tf.repeat
        # query.size(-3) is num_heads, key.size(-3) is num_kv_heads
        key = tf.repeat(key, query.shape[-3] // key.shape[-3], axis=-3)
        value = tf.repeat(value, query.shape[-3] // value.shape[-3], axis=-3)

    # Conversion: @ -> tf.matmul, transpose(-2, -1) -> transpose_b=True
    attn_weight = tf.matmul(query, key, transpose_b=True) * scale_factor
    attn_weight += attn_bias
    
    # Conversion: torch.softmax -> tf.nn.softmax
    attn_weight = tf.nn.softmax(attn_weight, axis=-1)
    
    # Conversion: torch.dropout(..., train=True) -> tf.nn.dropout
    # Note: TF dropout rate is probability of dropping, same as PyTorch p.
    # tf.nn.dropout is always active in eager execution, matching train=True.
    attn_weight = tf.nn.dropout(attn_weight, rate=dropout_p)
    
    # Conversion: @ -> tf.matmul
    return tf.matmul(attn_weight, value)
```