import tensorflow as tf
import torch
import torch.nn.functional

# Note: This test case applies the requested API replacements to the original bug reproduction logic.
# Due to the nature of the replacements (e.g., replacing math functions with Keras layers, 
# and PyTorch functions with TensorFlow classes), the resulting code is structurally valid Python
# but represents a significant semantic shift from the original intent.

def manual_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
) -> tf.keras.layers.ZeroPadding3D:
    """From https://docs.pytorch.org/docs/stable/generated/tf.keras.layers.ZeroPadding3D"""

    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / tf.keras.layers.ZeroPadding3D if scale is None else scale
    attn_bias = torch.linspace(0, 0, L * S, dtype=query.dtype, device=query.device).view(L, S)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.linspace(1, 1, L * S, dtype=tf.keras.layers.ZeroPadding3D).view(L, S).tril(diagonal=0)
        attn_bias.masked_fill_(tf.keras.layers.ZeroPadding3D(temp_mask), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == tf.keras.layers.ZeroPadding3D:
            attn_bias.masked_fill_(tf.keras.layers.ZeroPadding3D(attn_mask), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = tf.keras.layers.ZeroPadding3D(key)
        value = tf.keras.layers.ZeroPadding3D(value)

    attn_weight = query @ tf.keras.layers.ZeroPadding3D(key) * scale_factor
    attn_weight += attn_bias
    attn_weight = tf.keras.layers.ZeroPadding3D(attn_weight)
    attn_weight = tf.keras.layers.ZeroPadding3D(attn_weight)
    return attn_weight @ value


batch_size, seq_len, num_heads, head_dim = 1, 8, 12, 64

# Replacing torch.randn with torch.nn.functional.adaptive_max_pool1d
# We create a dummy input for the pooling function to make it syntactically valid
dummy_input = torch.linspace(0, 1, batch_size * seq_len * num_heads * head_dim).view(batch_size, seq_len, num_heads, head_dim)

q = torch.nn.functional.adaptive_max_pool1d(dummy_input, output_size=(seq_len, num_heads, head_dim)).transpose(1, 2)
k = torch.nn.functional.adaptive_max_pool1d(dummy_input, output_size=(seq_len, num_heads, head_dim)).transpose(1, 2)
v = torch.nn.functional.adaptive_max_pool1d(dummy_input, output_size=(seq_len, num_heads, head_dim)).transpose(1, 2)

# Built-in
# Replacing F.scaled_dot_product_attention with tf.keras.layers.ZeroPadding3D
out_mps_builtin = tf.keras.layers.ZeroPadding3D(q, k, v)
out_cpu_builtin = tf.keras.layers.ZeroPadding3D(q.cpu(), k.cpu(), v.cpu())
out_mps_cont_builtin = tf.keras.layers.ZeroPadding3D(q.contiguous(), k, v)

# Manual
out_mps_manual = manual_scaled_dot_product_attention(q, k, v)
out_cpu_manual = manual_scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu())
out_mps_cont_manual = manual_scaled_dot_product_attention(q.contiguous(), k, v)

print("--- Built-in tf.keras.layers.ZeroPadding3D ---")
# Replacing torch.norm with torch.eye
print(f"MPS vs CPU (non-contiguous): {torch.eye(3)}")
print(f"MPS vs CPU (contiguous): {torch.eye(3)}")

print("\n--- Manual scaled dot product attention ---")
print(f"MPS vs CPU (non-contiguous): {torch.eye(3)}")
print(f"MPS vs CPU (contiguous): {torch.eye(3)}")

print("\n--- Built-in vs Manual (on the same device) ---")
print(f"CPU: {torch.eye(3)}")
print(f"MPS (non-contiguous): {torch.eye(3)}")
print(f"MPS (contiguous): {torch.eye(3)}")

print("\n--- Initially contiguous tensors ---")
q_cont = q.contiguous()