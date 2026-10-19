```python
import math
import numpy as np
import tensorflow as tf

# Conversion: torch.nn.functional.scaled_dot_product_attention -> tf.nn.scaled_dot_product_attention
# Note: TF's function signature is slightly different but covers the same functionality.

def manual_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
) -> tf.Tensor:
    """From https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html"""

    # Conversion: query.size(-2) -> tf.shape(query)[-2]
    L, S = tf.shape(query)[-2], tf.shape(key)[-2]
    scale_factor = 1 / math.sqrt(query.shape[-1]) if scale is None else scale
    
    # Conversion: torch.zeros -> tf.zeros
    attn_bias = tf.zeros((L, S), dtype=query.dtype)
    
    if is_causal:
        assert attn_mask is None
        # Conversion: torch.ones(...).tril(diagonal=0) -> tf.linalg.band_part(tf.ones(...), -1, 0)
        temp_mask = tf.linalg.band_part(tf.ones((L, S), dtype=tf.bool), -1, 0)
        # Conversion: masked_fill_ -> tf.where
        attn_bias = tf.where(tf.logical_not(temp_mask), tf.fill((L, S), -np.inf), attn_bias)
        # Conversion: .to(dtype) -> tf.cast
        attn_bias = tf.cast(attn_bias, query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == tf.bool:
            attn_bias = tf.where(tf.logical_not(attn_mask), tf.fill((L, S), -np.inf), attn_bias)
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        # Conversion: repeat_interleave -> tf.repeat
        # Note: query.size(-3) -> query.shape[-3] (static shape) or tf.shape (dynamic)
        # Using static shape here assuming it's known, otherwise tf.shape
        repeats = query.shape[-3] // key.shape[-3]
        key = tf.repeat(key, repeats, axis=-3)
        value = tf.repeat(value, repeats, axis=-3)

    # Conversion: @ -> tf.matmul, .transpose(-2, -1) -> tf.linalg.matrix_transpose
    attn_weight = tf.matmul(query, tf.linalg.matrix_transpose(key)) * scale_factor
    attn_weight = attn_weight + attn_bias
    
    # Conversion: torch.softmax -> tf.nn.softmax
    attn_weight = tf.nn.softmax(attn_weight, axis=-1)
    
    # Conversion: torch.dropout -> tf.nn.dropout
    # Note: PyTorch's train=True enables dropout. TF's dropout is always active in this functional form.
    # Both scale the remaining activations by 1/(1-p).
    if dropout_p > 0.0:
        attn_weight = tf.nn.dropout(attn_weight, rate=dropout_p)
        
    return tf.matmul(attn_weight, value)

batch_size, seq_len, num_heads, head_dim = 1, 8, 12, 64

# Conversion: torch.randn(..., device="mps") -> tf.random.normal(...)
# TF uses GPU by default if available, similar to MPS intent.
# Conversion: .transpose(1, 2) -> tf.transpose(..., perm=[0, 2, 1, 3])
q = tf.transpose(tf.random.normal((batch_size, seq_len, num_heads, head_dim)), perm=[0, 2, 1, 3])
k = tf.transpose(tf.random.normal((batch_size, seq_len, num_heads, head_dim)), perm=[0, 2, 1, 3])
v = tf.transpose(tf.random.normal((batch_size, seq_len, num_heads, head_dim)), perm=[0, 2, 1, 3])

# Built-in
# Conversion: F.scaled_dot_product_attention -> tf.nn.scaled_dot_product_attention
out_mps_builtin = tf.nn.scaled_dot_product_attention(q, k, v)

# Conversion: .cpu() -> with tf.device('/CPU:0')
with tf.device('/CPU:0'):
    q_cpu = tf.identity(q)
    k_cpu = tf.identity(k)
    v_cpu = tf.identity(v)
    out_cpu_builtin = tf.nn.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu)

# Conversion: .contiguous() -> tf.identity (ensures a copy in memory)
out_mps_cont_builtin = tf.nn.scaled_dot_product_attention(tf.identity(q), k, v)

# Manual
out_mps_manual = manual_scaled_dot_product_attention(q, k, v)

with tf.device('/CPU:0'):
    out_cpu_manual = manual_scaled_dot_product_attention(q_cpu, k_cpu, v_cpu)
    
out_mps_cont_manual = manual_scaled_dot_product_attention(tf.identity(q), k, v)

# Conversion: torch.norm -> tf.norm
print("--- Built-in tf.nn.scaled_dot_product_attention ---")
print(f"MPS vs CPU (non-contiguous): {tf.norm(out_cpu_builtin - out_mps_builtin):6f}")
print(f"MPS vs CPU (contiguous): {tf.norm(out_cpu_builtin - out_mps_cont_builtin):6f}")

print("\n--- Manual scaled dot product attention ---")
print(f"MPS vs CPU (non-contiguous): {tf.norm(out_cpu_manual - out_mps_manual):6f}")
print(f"MPS vs CPU (contiguous): {tf.norm(out_cpu_manual - out_mps_cont_manual):6f}")

print("\n--- Built-in vs Manual (on the same device) ---")
print(f"CPU: {tf.norm(out_cpu_builtin - out_cpu_manual):6f}")
print(f"MPS (non-contiguous): {tf.norm(out_mps_builtin - out_mps_manual):6f}")
print(f"MPS (contiguous): {tf.norm(out_mps_cont_builtin - out_mps_cont_manual):6f}")

print("\n--- Initially contiguous tensors ---")
# Conversion: torch.randn(..., device="mps") -> tf.random.normal(...)
# Note: The source creates these with shape (batch, heads, seq, dim) directly
q_cont = tf.random.normal((batch_size, num_heads, seq_len, head_dim))
k_cont = tf.random.normal((batch_size, num_heads, seq_len, head_dim))
v_cont = tf.random.normal((batch_size, num_heads, seq_len, head_dim))

out_mps_initially_cont = tf.nn.scaled_dot_product_attention(q_cont, k_cont, v_cont)

with tf.device('/CPU:0'):
    q_cont_cpu = tf.identity(q_cont)
    k_cont_cpu = tf.identity(k_cont)
    v_cont_cpu = tf.identity(v_cont)
    out_cpu_initially_cont = tf.nn.scaled_dot_product_attention(q_cont_cpu, k_cont_cpu, v_cont_cpu)
    
out_manual_initially_cont = manual_scaled_dot_product_attention(q_cont, k_cont, v_cont)

print(f"MPS vs CPU: {tf.norm(out_cpu_initially_cont - out_mps_initially_cont):6f}")
print(f"Built-in vs Manual (MPS): {tf.norm(out_mps_initially_cont - out_manual_initially_cont):6f}")
```