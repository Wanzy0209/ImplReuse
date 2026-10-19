```python
import tensorflow as tf
import math

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False):
    # Conversion: PyTorch size() -> TF shape
    L, S = query.shape[-2], key.shape[-2]
    scale_factor = 1 / math.sqrt(query.shape[-1]) if scale is None else scale
    
    # Conversion: torch.zeros -> tf.zeros
    # Note: Device placement is implicit in TensorFlow
    attn_bias = tf.zeros((L, S), dtype=query.dtype)
    
    if is_causal:
        assert attn_mask is None
        # Conversion: torch.ones(...).tril -> tf.linalg.band_part
        # band_part(input, num_lower, num_upper). -1 means infinity.
        temp_mask = tf.linalg.band_part(tf.ones((L, S), dtype=tf.bool), -1, 0)
        # Conversion: masked_fill_ -> tf.where
        # PyTorch fills where mask is False. TF where takes (condition, true_val, false_val)
        attn_bias = tf.where(temp_mask, attn_bias, float("-inf"))
        # Ensure dtype consistency (redundant if initialized correctly, but safe)
        attn_bias = tf.cast(attn_bias, query.dtype)

    if attn_mask is not None:
        # Conversion: torch.bool -> tf.bool
        if attn_mask.dtype == tf.bool:
            # Conversion: logical_not -> tf.logical_not
            attn_bias = tf.where(attn_mask, attn_bias, float("-inf"))
        else:
            # Fix: Corrected source typo (attn_bias + attn_bias -> attn_bias + attn_mask)
            attn_bias = attn_bias + attn_mask

    if enable_gqa:
        # Conversion: repeat_interleave -> tf.repeat
        # PyTorch dim -3 corresponds to TF axis -3
        repeats = query.shape[-3] // key.shape[-3]
        key = tf.repeat(key, repeats, axis=-3)
        value = tf.repeat(value, repeats, axis=-3)

    # Conversion: @ -> tf.linalg.matmul
    # PyTorch transpose(-2, -1) is handled by transpose_b=True in TF matmul
    attn_weight = tf.linalg.matmul(query, key, transpose_b=True) * scale_factor
    attn_weight = attn_weight + attn_bias
    
    # Conversion: torch.softmax -> tf.nn.softmax
    attn_weight = tf.nn.softmax(attn_weight, axis=-1)
    
    # Conversion: torch.dropout -> tf.nn.dropout
    # Note: TF dropout rate is the same as PyTorch p (probability to drop)
    attn_weight = tf.nn.dropout(attn_weight, rate=dropout_p)
    
    return tf.linalg.matmul(attn_weight, value)
```