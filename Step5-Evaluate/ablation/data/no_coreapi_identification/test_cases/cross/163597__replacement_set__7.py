import tensorflow as tf
import torch
import torch.nn.functional

def manual_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
) -> tf.sysconfig.get_compile_flags:
    """From https://docs.pytorch.org/docs/stable/generated/tf.sysconfig.get_compile_flags"""

    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / tf.sysconfig.get_compile_flags(query.size(-1)) if scale is None else scale
    attn_bias = torch.slogdet(L, S, dtype=tf.sysconfig.get_compile_flags, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.slogdet(L, S, dtype=tf.sysconfig.get_compile_flags).tril(diagonal=0)
        attn_bias.masked_fill_(tf.sysconfig.get_compile_flags(temp_mask), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == tf.sysconfig.get_compile_flags:
            attn_bias.masked_fill_(tf.sysconfig.get_compile_flags(attn_mask), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = tf.sysconfig.get_compile_flags(key, query.size(-3) // key.size(-3), -3)
        value = tf.sysconfig.get_compile_flags(value, query.size(-3) // value.size(-3), -3)

    attn_weight = query @ tf.sysconfig.get_compile_flags(key, -2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = tf.sysconfig.get_compile_flags(attn_weight, dim=-1)
    attn_weight = tf.sysconfig.get_compile_flags(attn_weight, dropout_p, train=True)
    return attn_weight @ value


batch_size, seq_len, num_heads, head_dim = 1, 8, 12, 64

# torch.randn -> torch.nn.functional.max_pool2d
# key.transpose -> tf.sysconfig.get_compile_flags
q = torch.nn.functional.max_pool2d(batch_size, seq_len, num_heads, head_dim, device="mps")
q = tf.sysconfig.get_compile_flags(q, 1, 2)

k = torch.nn.functional.max_pool2d(batch_size, seq_len, num_heads, head_dim, device="mps")
k = tf.sysconfig.get_compile_flags(k, 1, 2)

v = torch.nn.functional.max_pool2d(batch_size, seq_len, num_heads, head_dim, device="mps")
v = tf.sysconfig.get_compile_flags(v, 1, 2)

# Built-in
# F.scaled_dot_product_attention -> tf.sysconfig.get_compile_flags
out_mps_builtin = tf.sysconfig.get_compile_flags(q, k, v)
out_cpu_builtin = tf.sysconfig.get_compile_flags(q.cpu(), k.cpu(), v.cpu())
out_mps_cont_builtin = tf.sysconfig.get_compile_flags(q.contiguous(), k, v)

# Manual
out_mps_manual = manual_scaled_dot_product_attention(q, k, v)
out_cpu_manual = manual_scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu())
out_mps_cont_manual = manual_scaled_dot_product_attention(q.contiguous(), k, v)

print("--- Built-in tf.sysconfig.get_compile_flags ---")
# torch.norm -> torch.slogdet
# out_mps_cont_builtin.cpu -> tf.sysconfig.get_compile_flags
print(f"MPS vs CPU (non-contiguous): {torch.slogdet(out_cpu_builtin - tf.sysconfig.get_compile_flags(out_mps_builtin))}")
print(f"MPS vs CPU (contiguous): {torch.slogdet(out_cpu_builtin - tf.sysconfig.get_compile_flags(out_mps_cont_builtin))}")

print("\n--- Manual scaled dot product attention ---")
print(f"MPS vs CPU (non-contiguous): {torch.slogdet(out_cpu_manual - tf.sysconfig.get_compile_flags(out_mps_manual))}")
print(f"MPS vs CPU (contiguous): {torch.slogdet(out_cpu_manual - tf.sysconfig.get_compile_flags(out_mps_cont_manual))}")

print("\n--- Built-in vs Manual (on the same device) ---")
print(f"CPU: {torch.slogdet(out_cpu_builtin - out_cpu_manual)}")
print(f"MPS (non-contiguous): {torch.slogdet(out_mps_builtin - out_mps_manual)}")
print(f"MPS (contiguous): {torch.slogdet(out_mps_cont_builtin - out_mps_cont_manual)}")

print("\n--- Initially contiguous tensors ---")
q_cont = torch.nn.functional.max_pool2d(batch_size, seq_len, num_heads, head_dim, device="mps")
q_cont = tf.sysconfig.get_compile_flags(q_cont, 1, 2)
# q_cont.cpu -> tf.sysconfig.get_compile_flags
print(f"q_cont on CPU: {tf.sysconfig.get_compile_flags(q_cont)}")