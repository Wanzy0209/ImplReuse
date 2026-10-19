```python
import math

import tensorflow as tf


def manual_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
) -> tf.Tensor:
    """From https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html"""

    # Conversion: query.size(-2) -> tf.shape(query)[-2]
    L, S = tf.shape(query)[-2], tf.shape(key)[-2]
    
    # Conversion: query.size(-1) -> query.shape[-1] (static shape preferred for sqrt if possible, else dynamic)
    scale_factor = 1 / math.sqrt(float(query.shape[-1])) if scale is None else scale
    
    # Conversion: torch.zeros -> tf.zeros
    attn_bias = tf.zeros((L, S), dtype=query.dtype)
    
    if is_causal:
        # Conversion: torch.ones(...).tril -> tf.linalg.band_part
        temp_mask = tf.linalg.band_part(tf.ones((L, S), dtype=tf.bool), -1, 0)
        # Conversion: masked_fill_ -> tf.where
        attn_bias = tf.where(tf.logical_not(temp_mask), float("-inf"), attn_bias)
        # Conversion: .to(dtype) -> tf.cast
        attn_bias = tf.cast(attn_bias, query.dtype)

    if attn_mask is not None:
        # Conversion: torch.bool -> tf.bool
        if attn_mask.dtype == tf.bool:
            attn_bias = tf.where(tf.logical_not(attn_mask), float("-inf"), attn_bias)
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        # Conversion: repeat_interleave -> tf.repeat
        # Note: axis -3 corresponds to the num_heads dimension in (B, H, L, D)
        key = tf.repeat(key, query.shape[-3] // key.shape[-3], axis=-3)
        value = tf.repeat(value, query.shape[-3] // value.shape[-3], axis=-3)

    # Conversion: @ -> tf.matmul, transpose(-2, -1) -> tf.transpose(..., perm=[0, 1, 3, 2])
    attn_weight = tf.matmul(query, tf.transpose(key, perm=[0, 1, 3, 2])) * scale_factor
    attn_weight = attn_weight + attn_bias
    
    # Conversion: torch.softmax(..., dim=-1) -> tf.nn.softmax(..., axis=-1)
    attn_weight = tf.nn.softmax(attn_weight, axis=-1)
    
    # Conversion: torch.dropout(..., train=True) -> tf.nn.dropout(...)
    # Note: TF dropout rate is probability of dropping, same as PyTorch p
    attn_weight = tf.nn.dropout(attn_weight, rate=dropout_p)
    
    return tf.matmul(attn_weight, value)


batch_size, seq_len, num_heads, head_dim = 1, 8, 12, 64

# Conversion: device="mps" -> tf.device('/GPU:0') (assuming GPU available for MPS equivalent)
# We use a context manager to place tensors on the GPU
device_name = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

with tf.device(device_name):
    # Conversion: torch.randn -> tf.random.normal
    # Conversion: .transpose(1, 2) -> tf.transpose(..., perm=[0, 2, 1, 3])
    q = tf.transpose(tf.random.normal((batch_size, seq_len, num_heads, head_dim)), perm=[0, 2, 1, 3])
    k = tf.transpose(tf.random.normal((batch_size, seq_len, num_heads, head_dim)), perm=[0, 2, 1, 3])
    v = tf.transpose(tf.random.normal((batch_size, seq_len, num_heads, head_dim)), perm=[0, 2, 1, 3])

    # Built-in
    # Conversion: F.scaled_dot_product_attention -> tf.nn.scaled_dot_product_attention
    out_mps_builtin = tf.nn.scaled_dot_product_attention(q, k, v)
    
    # Conversion: .cpu() -> tf.identity inside CPU device context
    with tf.device('/CPU:0'):
        q_cpu = tf.identity(q)
        k_cpu = tf.identity(k)
        v_cpu = tf.identity(v)
    
    out_cpu_builtin = tf.nn.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu)
    
    # Conversion: .contiguous() -> tf.identity (no-op but ensures distinct tensor)
    out_mps_cont_builtin = tf.nn.scaled_dot_product_attention(tf.identity(q), k, v)

    # Manual
    out_mps_manual = manual_scaled_dot_product_attention(q, k, v)
    out_cpu_manual = manual_scaled_dot_product_attention(q_cpu, k_cpu, v_cpu)
    out_mps_cont_manual = manual_scaled_dot_product_attention(tf.identity(q), k, v)

print("--- Built-in F.scaled_dot_product_attention ---")
# Conversion: torch.norm -> tf.norm
# Conversion: .cpu() -> tf.identity on CPU
with tf.device('/CPU:0'):
    diff_mps_cpu = tf.norm(out_cpu_builtin - tf.identity(out_mps_builtin))
    diff_mps_cont_cpu = tf.norm(out_cpu_builtin - tf.identity(out_mps_cont_builtin))

print(f"MPS vs CPU (non-contiguous): {diff_mps_cpu:6f}")
print(f"MPS vs CPU (contiguous): {diff_mps_cont_cpu:6f}")

print("\n--- Manual scaled dot product attention ---")
with tf.device('/CPU:0'):
    diff_mps_cpu_man = tf.norm(out_cpu_manual - tf.identity(out_mps_manual))
    diff_mps_cont_cpu_man = tf.norm(out_cpu_manual - tf.identity(out_mps_cont_manual))

print(f"MPS vs CPU (non-contiguous): {diff_mps_cpu_man:6f}")
print(f"MPS vs CPU (contiguous): {diff_mps_cont_cpu_man:6f}")

print("\n--- Built-in vs Manual (on the same device) ---")
# Note: out_cpu_builtin and out_cpu_manual are already on CPU
print(f"CPU: {tf.norm(out_cpu_builtin - out_cpu_manual):6f}")
# out_mps_builtin and out_mps_manual are on GPU
print(f"MPS (non-contiguous): {tf.norm(out_mps_builtin - out_mps_manual):6f}")
print(f"MPS (contiguous): {tf.norm(out_mps_cont_builtin - out_mps_cont_manual):6f}")

print("\n--- Initially contiguous tensors ---")
with tf.device(device_name):
    # Conversion: torch.randn(..., device="mps") -> tf.random.normal inside device context
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

with tf.device('/CPU:0'):
    diff_init_cont = tf.norm(out_cpu_initially_cont - tf.identity(out_mps_initially_cont))
    diff_builtin_man = tf.norm(out_mps_initially_cont - tf.identity(out_manual_initially_cont))

print(f"MPS vs CPU: {diff_init_cont:6f}")
print(f"Built-in vs Manual (MPS): {diff_builtin_man:6f}")
```