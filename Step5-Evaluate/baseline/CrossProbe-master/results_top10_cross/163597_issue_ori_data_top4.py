```python
import math

import tensorflow as tf


def manual_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
) -> tf.Tensor:
    """From https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html"""

    L, S = query.shape[-2], key.shape[-2]
    scale_factor = 1 / math.sqrt(query.shape[-1]) if scale is None else scale
    attn_bias = tf.zeros((L, S), dtype=query.dtype)
    
    if is_causal:
        assert attn_mask is None
        # Conversion: torch.tril -> tf.linalg.band_part
        temp_mask = tf.linalg.band_part(tf.ones((L, S), dtype=tf.bool), -1, 0)
        # Conversion: masked_fill with logical_not -> tf.where
        attn_bias = tf.where(temp_mask, attn_bias, tf.cast(float("-inf"), attn_bias.dtype))

    if attn_mask is not None:
        if attn_mask.dtype == tf.bool:
            attn_bias = tf.where(attn_mask, attn_bias, tf.cast(float("-inf"), attn_bias.dtype))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        # Conversion: repeat_interleave -> tf.repeat
        key = tf.repeat(key, repeats=query.shape[-3] // key.shape[-3], axis=-3)
        value = tf.repeat(value, repeats=query.shape[-3] // value.shape[-3], axis=-3)

    # Conversion: @ -> tf.matmul, transpose(-2, -1) -> transpose_b=True
    attn_weight = tf.matmul(query, key, transpose_b=True) * scale_factor
    attn_weight += attn_bias
    # Conversion: torch.softmax -> tf.nn.softmax
    attn_weight = tf.nn.softmax(attn_weight, axis=-1)
    # Conversion: torch.dropout -> tf.nn.dropout
    # Note: PyTorch dropout_p is prob of keeping, TF rate is prob of dropping.
    # However, standard definition usually aligns p=0.1 to 10% dropout.
    # PyTorch: dropout_p=0.0 means no dropout. TF: rate=0.0 means no dropout.
    # PyTorch: dropout_p=0.1 means 10% elements zeroed. TF: rate=0.1 means 10% elements zeroed.
    attn_weight = tf.nn.dropout(attn_weight, rate=dropout_p)
    return tf.matmul(attn_weight, value)


batch_size, seq_len, num_heads, head_dim = 1, 8, 12, 64

# Conversion: Determine device (MPS equivalent is GPU in TF)
device_name = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

with tf.device(device_name):
    # Conversion: torch.randn -> tf.random.normal
    # Conversion: .transpose(1, 2) -> tf.transpose(..., perm=[0, 2, 1, 3])
    q = tf.random.normal((batch_size, seq_len, num_heads, head_dim))
    q = tf.transpose(q, perm=[0, 2, 1, 3])
    k = tf.random.normal((batch_size, seq_len, num_heads, head_dim))
    k = tf.transpose(k, perm=[0, 2, 1, 3])
    v = tf.random.normal((batch_size, seq_len, num_heads, head_dim))
    v = tf.transpose(v, perm=[0, 2, 1, 3])

# Built-in
# Conversion: F.scaled_dot_product_attention -> tf.nn.scaled_dot_product_attention
# Note: TF SDPA expects (B, T, S, D) or (B, H, T, S, D) with attention_axes.
# Here inputs are (B, H, S, D), so we specify attention_axes=[2] (the sequence dimension).
out_mps_builtin = tf.nn.scaled_dot_product_attention(q, k, v, attention_axes=[2])

# Conversion: .cpu() -> tf.identity inside CPU device block
with tf.device('/CPU:0'):
    q_cpu = tf.identity(q)
    k_cpu = tf.identity(k)
    v_cpu = tf.identity(v)

out_cpu_builtin = tf.nn.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu, attention_axes=[2])

# Conversion: .contiguous() -> tf.identity (forces a copy)
out_mps_cont_builtin = tf.nn.scaled_dot_product_attention(tf.identity(q), k, v, attention_axes=[2])

# Manual
out_mps_manual = manual_scaled_dot_product_attention(q, k, v)
out_cpu_manual = manual_scaled_dot_product_attention(q_cpu, k_cpu, v_cpu)
out_mps_cont_manual = manual_scaled_dot_product_attention(tf.identity(q), k, v)

print("--- Built-in F.scaled_dot_product_attention ---")
# Conversion: torch.norm -> tf.norm
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
with tf.device(device_name):
    q_cont = tf.random.normal((batch_size, num_heads, seq_len, head_dim))
    k_cont = tf.random.normal((batch_size, num_heads, seq_len, head_dim))
    v_cont = tf.random.normal((batch_size, num_heads, seq_len, head_dim))

out_mps_initially_cont = tf.nn.scaled_dot_product_attention(q_cont, k_cont, v_cont, attention_axes=[2])

with tf.device('/CPU:0'):
    q_cont_cpu = tf.identity(q_cont)
    k_cont_cpu = tf.identity(k_cont)
    v_cont_cpu = tf.identity(v_cont)

out_cpu_initially_cont = tf.nn.scaled_dot_product_attention(q_cont_cpu, k_cont_cpu, v_cont_cpu, attention_axes=[2])
out_manual_initially_cont = manual_scaled_dot_product_attention(q_cont, k_cont, v_cont)

print(f"MPS vs CPU: {tf.norm(out_cpu_initially_cont - out_mps_initially_cont):6f}")
print(f"Built-in vs Manual (MPS): {tf.norm(out_mps_initially_cont - out_manual_initially_cont):6f}")
```