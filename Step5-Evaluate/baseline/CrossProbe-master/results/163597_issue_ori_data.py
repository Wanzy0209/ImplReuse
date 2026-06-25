```python
import math

import tensorflow as tf


def manual_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
) -> tf.Tensor:
    """From https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html"""

    # Conversion: query.size(-2) -> tf.shape(query)[-2]
    L, S = tf.shape(query)[-2], tf.shape(key)[-2]
    
    # Conversion: query.size(-1) -> tf.shape(query)[-1], cast to float for sqrt
    scale_factor = 1.0 / math.sqrt(tf.cast(tf.shape(query)[-1], tf.float32)) if scale is None else scale
    
    # Conversion: torch.zeros -> tf.zeros
    attn_bias = tf.zeros((L, S), dtype=query.dtype)
    
    if is_causal:
        # Conversion: torch.ones(...).tril -> tf.linalg.band_part
        # band_part(..., -1, 0) returns lower triangular matrix
        temp_mask = tf.linalg.band_part(tf.ones((L, S), dtype=tf.bool), -1, 0)
        
        # Conversion: masked_fill_ -> tf.where
        attn_bias = tf.where(
            tf.logical_not(temp_mask), 
            tf.cast(float("-inf"), attn_bias.dtype), 
            attn_bias
        )

    if attn_mask is not None:
        if attn_mask.dtype == tf.bool:
            attn_bias = tf.where(
                tf.logical_not(attn_mask), 
                tf.cast(float("-inf"), attn_bias.dtype), 
                attn_bias
            )
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        # Conversion: repeat_interleave -> tf.repeat
        # query.size(-3) corresponds to the head dimension (axis 1 in [B, H, L, D])
        repeats = tf.shape(query)[-3] // tf.shape(key)[-3]
        key = tf.repeat(key, repeats=repeats, axis=1)
        value = tf.repeat(value, repeats=repeats, axis=1)

    # Conversion: @ -> tf.matmul, transpose(-2, -1) -> tf.transpose(..., [0, 1, 3, 2])
    # Input shape: [B, H, L, D], Key shape: [B, H, S, D]
    # Transpose Key to [B, H, D, S] for matmul
    attn_weight = tf.matmul(query, tf.transpose(key, perm=[0, 1, 3, 2])) * scale_factor
    attn_weight = attn_weight + attn_bias
    
    # Conversion: torch.softmax(..., dim=-1) -> tf.nn.softmax(..., axis=-1)
    attn_weight = tf.nn.softmax(attn_weight, axis=-1)
    
    # Conversion: torch.dropout(..., p, train=True) -> tf.nn.dropout(..., rate=1-p)
    # PyTorch p is probability of keeping, TF rate is probability of dropping
    attn_weight = tf.nn.dropout(attn_weight, rate=1.0 - dropout_p)
    
    return tf.matmul(attn_weight, value)


def tf_builtin_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
    """
    Placeholder for the 'Built-in' behavior in TensorFlow.
    TF does not have a direct functional equivalent to F.scaled_dot_product_attention 
    that takes pre-projected Q, K, V tensors. We use standard TF ops here.
    """
    L, S = tf.shape(query)[-2], tf.shape(key)[-2]
    scale_factor = 1.0 / math.sqrt(tf.cast(tf.shape(query)[-1], tf.float32))
    
    attn_weight = tf.matmul(query, tf.transpose(key, perm=[0, 1, 3, 2])) * scale_factor

    if is_causal:
        # Create causal mask: 0 for upper triangle, 1 for lower
        mask = tf.linalg.band_part(tf.ones((L, S), dtype=query.dtype), -1, 0)
        # Apply -inf to upper triangle
        causal_mask = (1.0 - mask) * -1e9
        attn_weight = attn_weight + causal_mask

    if attn_mask is not None:
        if attn_mask.dtype == tf.bool:
            attn_weight = tf.where(
                tf.logical_not(attn_mask), 
                tf.cast(float("-inf"), attn_weight.dtype), 
                attn_weight
            )
        else:
            attn_weight = attn_weight + attn_mask

    attn_weight = tf.nn.softmax(attn_weight, axis=-1)
    attn_weight = tf.nn.dropout(attn_weight, rate=1.0 - dropout_p)
    return tf.matmul(attn_weight, value)


batch_size, seq_len, num_heads, head_dim = 1, 8, 12, 64

# Conversion: torch.randn -> tf.random.normal
# Conversion: device="mps" -> Default device (GPU if available)
# Conversion: .transpose(1, 2) -> tf.transpose(..., [0, 2, 1, 3])
q = tf.random.normal((batch_size, seq_len, num_heads, head_dim))
q = tf.transpose(q, [0, 2, 1, 3])

k = tf.random.normal((batch_size, seq_len, num_heads, head_dim))
k = tf.transpose(k, [0, 2, 1, 3])

v = tf.random.normal((batch_size, seq_len, num_heads, head_dim))
v = tf.transpose(v, [0, 2, 1, 3])

# Conversion: .cpu() -> tf.device('/CPU:0')
with tf.device('/CPU:0'):
    q_cpu = tf.identity(q)
    k_cpu = tf.identity(k)
    v_cpu = tf.identity(v)

# Built-in
out_gpu_builtin = tf_builtin_scaled_dot_product_attention(q, k, v)
out_cpu_builtin = tf_builtin_scaled_dot_product_attention(q_cpu, k_cpu, v_cpu)
# Conversion: .contiguous() -> No-op in TF (tensors are row-major)
out_gpu_cont_builtin = tf_builtin_scaled_dot_product_attention(q, k, v)

# Manual
out_gpu_manual = manual_scaled_dot_product_attention(q, k, v)
out_cpu_manual = manual_scaled_dot_product_attention(q_cpu, k_cpu, v_cpu)
out_gpu_cont_manual = manual_scaled_dot_product_attention(q, k, v)

# Conversion: torch.norm -> tf.norm
print("--- Built-in tf_builtin_scaled_dot_product_attention ---")
print(f"GPU vs CPU (non-contiguous): {tf.norm(out_cpu_builtin - out_gpu_builtin):6f}")
print(f"GPU vs CPU (contiguous): {tf.norm(out_cpu_builtin - out_gpu_cont_builtin):6f}")

print("\n--- Manual scaled dot product attention ---")
print(f"GPU vs CPU (non-contiguous): {tf.norm(out_cpu_manual - out_gpu_manual):6f}")
print(f"GPU vs CPU (contiguous): {tf.norm(out_cpu_manual - out_gpu_cont_manual):6f}")

print("\n--- Built-in vs Manual (on the same device) ---")
print(f"CPU: {tf.norm(out_cpu_builtin - out_cpu_manual):6f}")
print(f"GPU (non-contiguous): {tf.norm(out_gpu_builtin - out_gpu_manual):6f}")
print(f"GPU (contiguous): {tf.norm(out_gpu_cont_builtin - out_gpu_cont_manual):6f}")

print("\n--- Initially contiguous tensors ---")
# Create tensors already in the transposed shape
q_cont = tf.random.normal((batch_size, num_heads, seq_len, head_dim))
k_cont = tf.random.normal((batch_size, num_heads, seq_len, head_dim))
v_cont = tf.random.normal((batch_size, num_heads, seq_len, head_dim))

out_gpu_initially_cont = tf_builtin_scaled_dot_product_attention(q_cont, k_cont, v_cont)

with tf.device('/CPU:0'):
    q_cont_cpu = tf.identity(q_cont)
    k_cont_cpu = tf.identity(k_cont)
    v_cont_cpu = tf.identity(v_cont)

out_cpu_initially_cont = tf_builtin_scaled_dot_product_attention(q_cont_cpu, k_cont_cpu, v_cont_cpu)
out_manual_initially_cont = manual_scaled_dot_product_attention(q_cont, k_cont, v_cont)

print(f"GPU vs CPU: {tf.norm(out_cpu_initially_cont - out_gpu_initially_cont):6f}")
print(f"Built-in vs Manual (GPU): {tf.norm(out_gpu_initially_cont - out_manual_initially_cont):6f}")
```