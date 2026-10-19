```python
import math
import tensorflow as tf

# Note: TensorFlow does not have a direct "mps" device string.
# We map "mps" to GPU ('/GPU:0') for this translation.
# If no GPU is available, this code may raise an error unless handled.
# We assume a GPU is available to match the original script's intent.

def manual_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
):
    """Manual implementation of scaled dot product attention in TensorFlow."""
    # Get shapes
    L, S = tf.shape(query)[-2], tf.shape(key)[-2]
    
    # Calculate scale factor
    if scale is None:
        # Cast to float for sqrt operation
        scale_factor = 1.0 / tf.math.sqrt(tf.cast(query.shape[-1], query.dtype))
    else:
        scale_factor = scale

    # Initialize attention bias
    attn_bias = tf.zeros((L, S), dtype=query.dtype)

    if is_causal:
        # Create causal mask (lower triangular)
        # tf.linalg.band_part(input, -1, 0) returns the lower triangular part
        mask_shape = (L, S)
        temp_mask = tf.linalg.band_part(tf.ones(mask_shape, dtype=tf.bool), -1, 0)
        # Fill upper triangle with -inf
        attn_bias = tf.where(temp_mask, attn_bias, tf.cast(-float('inf'), attn_bias.dtype))

    if attn_mask is not None:
        if attn_mask.dtype == tf.bool:
            # Where mask is False, fill with -inf
            attn_bias = tf.where(attn_mask, attn_bias, tf.cast(-float('inf'), attn_bias.dtype))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        # Grouped Query Attention: repeat keys and values
        # query.size(-3) // key.size(-3)
        repeats = query.shape[-3] // key.shape[-3]
        key = tf.repeat(key, repeats=repeats, axis=-3)
        value = tf.repeat(value, repeats=repeats, axis=-3)

    # Calculate attention scores
    # query @ key.transpose(-2, -1)
    attn_weight = tf.matmul(query, key, transpose_b=True) * scale_factor
    attn_weight = attn_weight + attn_bias
    
    # Softmax
    attn_weight = tf.nn.softmax(attn_weight, axis=-1)
    
    # Dropout
    # Note: tf.nn.dropout uses 'rate' (probability to drop), same as PyTorch's dropout_p
    attn_weight = tf.nn.dropout(attn_weight, rate=dropout_p)
    
    # Return weighted sum
    return tf.matmul(attn_weight, value)

# Helper to simulate .cpu() in PyTorch
def to_cpu(tensor):
    with tf.device('/CPU:0'):
        return tf.identity(tensor)

# Helper to simulate .contiguous() in PyTorch
# In TF, tensors are immutable, so operations return new tensors.
# tf.identity creates a copy, which is roughly equivalent to ensuring contiguous memory.
def make_contiguous(tensor):
    return tf.identity(tensor)

batch_size, seq_len, num_heads, head_dim = 1, 8, 12, 64

# Create tensors on GPU (simulating MPS)
with tf.device('/GPU:0'):
    # PyTorch: randn(...) -> TF: random.normal
    # PyTorch: transpose(1, 2) -> TF: transpose
    q = tf.transpose(tf.random.normal((batch_size, seq_len, num_heads, head_dim)), perm=[0, 2, 1, 3])
    k = tf.transpose(tf.random.normal((batch_size, seq_len, num_heads, head_dim)), perm=[0, 2, 1, 3])
    v = tf.transpose(tf.random.normal((batch_size, seq_len, num_heads, head_dim)), perm=[0, 2, 1, 3])

# Built-in
# PyTorch: F.scaled_dot_product_attention
# TensorFlow: tf.nn.scaled_dot_product_attention
with tf.device('/GPU:0'):
    out_mps_builtin = tf.nn.scaled_dot_product_attention(q, k, v)
    out_mps_cont_builtin = tf.nn.scaled_dot_product_attention(make_contiguous(q), k, v)

# CPU versions
q_cpu = to_cpu(q)
k_cpu = to_cpu(k)
v_cpu = to_cpu(v)
out_cpu_builtin = tf.nn.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu)

# Manual
with tf.device('/GPU:0'):
    out_mps_manual = manual_scaled_dot_product_attention(q, k, v)
    out_mps_cont_manual = manual_scaled_dot_product_attention(make_contiguous(q), k, v)

out_cpu_manual = manual_scaled_dot_product_attention(q_cpu, k_cpu, v_cpu)

# Print results
# Using .numpy() to get values for printing
print("--- Built-in tf.nn.scaled_dot_product_attention ---")
# torch.norm -> tf.norm
print(f"MPS vs CPU (non-contiguous): {tf.norm(out_cpu_builtin - to_cpu(out_mps_builtin)).numpy():6f}")
print(f"MPS vs CPU (contiguous): {tf.norm(out_cpu_builtin - to_cpu(out_mps_cont_builtin)).numpy():6f}")

print("\n--- Manual scaled dot product attention ---")
print(f"MPS vs CPU (non-contiguous): {tf.norm(out_cpu_manual - to_cpu(out_mps_manual)).numpy():6f}")
print(f"MPS vs CPU (contiguous): {tf.norm(out_cpu_manual - to_cpu(out_mps_cont_manual)).numpy():6f}")

print("\n--- Built-in vs Manual (on the same device) ---")
print(f"CPU: {tf.norm(out_cpu_builtin - out_cpu_manual).numpy():6f}")
print(f"MPS (non-contiguous): {tf.norm(out_mps_builtin - out_mps_manual).numpy():6f}")
print(f"MPS (contiguous): {tf.norm(out_mps_cont_builtin - out_mps_cont_manual).numpy():6f}")

print("\n--- Initially contiguous tensors ---")
with tf.device('/GPU:0'):
    q_cont = tf.random.normal((batch_size, num_heads, seq_len, head_dim))
    k_cont = tf.random.normal((batch_size, num_heads, seq_len, head_dim))
    v_cont = tf.random.normal((batch_size, num_heads, seq_len, head_dim))

out_mps_initially_cont = tf.nn.scaled_dot_product_attention(q_cont, k_cont, v_cont)
out_cpu_initially_cont = tf.nn.scaled_dot_product_attention(to_cpu(q_cont), to_cpu(k_cont), to_cpu(v_cont))
out_manual_initially_cont = manual_scaled_dot_product_attention(q_cont, k_cont, v_cont)

print(f"MPS vs CPU: {tf.norm(out_cpu_initially_cont - to_cpu(out_mps_initially_cont)).numpy():6f}")
print(f"Built-in vs Manual (MPS): {tf.norm(out_mps_initially_cont -