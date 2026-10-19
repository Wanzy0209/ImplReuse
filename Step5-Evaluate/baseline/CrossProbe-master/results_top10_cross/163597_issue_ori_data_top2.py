```python
import tensorflow as tf

# Conversion: math is not strictly needed as we use tf.sqrt for tensor operations
# import math 

def manual_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
) -> tf.Tensor:
    """From https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html"""

    # Conversion: query.size(-2) -> tf.shape(query)[-2]
    L, S = tf.shape(query)[-2], tf.shape(key)[-2]
    
    # Conversion: math.sqrt(query.size(-1)) -> tf.sqrt(tf.cast(...))
    # Note: query.size(-1) returns a Tensor in TF, so we use tf.sqrt
    scale_factor = 1.0 / tf.sqrt(tf.cast(tf.shape(query)[-1], tf.float32)) if scale is None else scale
    
    # Conversion: torch.zeros(...) -> tf.zeros(...)
    attn_bias = tf.zeros((L, S), dtype=query.dtype)
    
    if is_causal:
        # assert attn_mask is None
        # Conversion: torch.ones(...).tril(...) -> tf.linalg.band_part(tf.ones(...), -1, 0)
        temp_mask = tf.ones((L, S), dtype=tf.bool)
        temp_mask = tf.linalg.band_part(temp_mask, -1, 0)
        
        # Conversion: masked_fill_ -> tf.where
        attn_bias = tf.where(tf.logical_not(temp_mask), float("-inf"), attn_bias)
        # Conversion: .to(query.dtype) -> tf.cast(..., query.dtype)
        attn_bias = tf.cast(attn_bias, query.dtype)

    if attn_mask is not None:
        # Conversion: attn_mask.dtype == torch.bool -> attn_mask.dtype == tf.bool
        if attn_mask.dtype == tf.bool:
            attn_bias = tf.where(tf.logical_not(attn_mask), float("-inf"), attn_bias)
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        # Conversion: repeat_interleave -> tf.repeat
        # query.size(-3) // key.size(-3)
        repeats = tf.shape(query)[-3] // tf.shape(key)[-3]
        key = tf.repeat(key, repeats=repeats, axis=-3)
        value = tf.repeat(value, repeats=repeats, axis=-3)

    # Conversion: @ -> tf.matmul, .transpose(-2, -1) -> tf.transpose(..., perm=[0, 1, 3, 2])
    attn_weight = tf.matmul(query, tf.transpose(key, perm=[0, 1, 3, 2])) * scale_factor
    attn_weight = attn_weight + attn_bias
    
    # Conversion: torch.softmax -> tf.nn.softmax
    attn_weight = tf.nn.softmax(attn_weight, axis=-1)
    
    # Conversion: torch.dropout -> tf.nn.dropout
    # Note: PyTorch dropout_p is prob of dropping, TF rate is prob of dropping.
    attn_weight = tf.nn.dropout(attn_weight, rate=dropout_p)
    
    return tf.matmul(attn_weight, value)


batch_size, seq_len, num_heads, head_dim = 1, 8, 12, 64

# Conversion: torch.randn -> tf.random.normal
# Conversion: device="mps" removed (TF handles device placement implicitly)
# Conversion: .transpose(1, 2) -> tf.transpose(..., perm=[0, 2, 1, 3])
q = tf.random.normal((batch_size, seq_len, num_heads, head_dim))
q = tf.transpose(q, perm=[0, 2, 1, 3])
k = tf.random.normal((batch_size, seq_len, num_heads, head_dim))
k = tf.transpose(k, perm=[0, 2, 1, 3])
v = tf.random.normal((batch_size, seq_len, num_heads, head_dim))
v = tf.transpose(v, perm=[0, 2, 1, 3])

# Built-in
# Conversion: F.scaled_dot_product_attention -> tf.nn.scaled_dot_product_attention
# Note: tf.nn.scaled_dot_product_attention is available in TF 2.11+
out_builtin = tf.nn.scaled_dot_product_attention(q, k, v)
# Conversion: .cpu() and .contiguous() are no-ops in TF or handled implicitly.
# We simplify the test to compare Built-in vs Manual.
out_manual = manual_scaled_dot_product_attention(q, k, v)

print("--- Built-in tf.nn.scaled_dot_product_attention ---")
# Conversion: torch.norm -> tf.norm
# Since we removed device splits, we just print a placeholder or check consistency
print(f"Built-in output norm: {tf.norm(out_builtin):6f}")

print("\n--- Manual scaled dot product attention ---")
print(f"Manual output norm: {tf.norm(out_manual):6f}")

print("\n--- Built-in vs Manual ---")
# Conversion: torch.norm -> tf.norm
print(f"Difference: {tf.norm(out_builtin - out_manual):6f}")

print("\n--- Initially contiguous tensors ---")
# Conversion: torch.randn -> tf.random.normal
# Note: In TF, tensors are not explicitly "contiguous" in the PyTorch sense.
q_cont = tf.random.normal((batch_size, num_heads, seq_len, head_dim))
k_cont = tf.random.normal((batch_size, num_heads, seq_len, head_dim))
v_cont = tf.random.normal((batch_size, num_heads, seq_len, head_dim))

out_builtin_cont = tf.nn.scaled_dot_product_attention(q_cont, k_cont, v_cont)
out_manual_cont = manual_scaled_dot_product_attention(q_cont, k_cont, v_cont)

print(f"Built-in vs Manual (contiguous): {tf.norm(out_builtin_cont - out_manual_cont):6f}")
```