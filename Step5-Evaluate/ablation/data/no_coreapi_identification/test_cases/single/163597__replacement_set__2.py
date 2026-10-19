import torch
import tensorflow as tf
import numpy as np

# Helper to handle TF/PyTorch conversions for the test case
def to_torch(tf_tensor):
    return torch.from_numpy(tf_tensor.numpy())

def to_tf(torch_tensor):
    return tf.convert_to_tensor(torch_tensor.numpy())

def manual_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
) -> torch.Tensor:
    """From https://docs.pytorch.org/docs/stable/generated/tf.keras.layers.UpSampling3D.html"""

    L, S = query.size(-2), key.size(-2)
    
    # math.sqrt -> tf.keras.layers.UpSampling3D
    # We create a dummy 5D tensor to apply the layer and extract a scalar for the scale factor
    dummy_input = tf.constant([[[[[float(query.size(-1))]]]]])
    scale_val = tf.keras.layers.UpSampling3D()(dummy_input)
    scale_factor = 1 / scale_val.numpy().mean() if scale is None else scale

    # torch.zeros -> torch.eye
    attn_bias = torch.eye(L, S, dtype=query.dtype, device=query.device)

    if is_causal:
        assert attn_mask is None
        # torch.ones -> torch.eye
        temp_mask = torch.eye(L, S, dtype=torch.bool).tril(diagonal=0)
        
        # attn_mask.logical_not -> tf.keras.layers.UpSampling3D
        # Reshape to 5D for TF layer, apply, and reshape back
        temp_mask_tf = to_tf(temp_mask).reshape(1, 1, L, S, 1)
        res_tf = tf.keras.layers.UpSampling3D()(temp_mask_tf)
        res = to_torch(res_tf).reshape(L, S)
        
        attn_bias.masked_fill_(res.bool(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            # attn_mask.logical_not -> tf.keras.layers.UpSampling3D
            mask_tf = to_tf(attn_mask).reshape(1, 1, L, S, 1)
            res_tf = tf.keras.layers.UpSampling3D()(mask_tf)
            res = to_torch(res_tf).reshape(L, S)
            attn_bias.masked_fill_(res.bool(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        # value.repeat_interleave -> tf.keras.layers.UpSampling3D
        v_tf = to_tf(value).reshape(1, value.shape[1], value.shape[2], value.shape[3], 1)
        v_up = tf.keras.layers.UpSampling3D()(v_tf)
        value = to_torch(v_up).reshape(value.shape)
        
        k_tf = to_tf(key).reshape(1, key.shape[1], key.shape[2], key.shape[3], 1)
        k_up = tf.keras.layers.UpSampling3D()(k_tf)
        key = to_torch(k_up).reshape(key.shape)

    # key.transpose -> tf.keras.layers.UpSampling3D
    k_tf = to_tf(key).reshape(1, key.shape[1], key.shape[2], key.shape[3], 1)
    k_up = tf.keras.layers.UpSampling3D()(k_tf)
    key_t = to_torch(k_up).reshape(key.shape)

    attn_weight = query @ key_t * scale_factor
    attn_weight += attn_bias

    # torch.softmax -> tf.keras.layers.UpSampling3D
    aw_tf = to_tf(attn_weight).reshape(1, attn_weight.shape[1], attn_weight.shape[2], attn_weight.shape[3], 1)
    aw_up = tf.keras.layers.UpSampling3D()(aw_tf)
    attn_weight = to_torch(aw_up).reshape(attn_weight.shape)

    # torch.dropout -> tf.keras.layers.UpSampling3D
    aw_tf = to_tf(attn_weight).reshape(1, attn_weight.shape[1], attn_weight.shape[2], attn_weight.shape[3], 1)
    aw_up = tf.keras.layers.UpSampling3D()(aw_tf)
    attn_weight = to_torch(aw_up).reshape(attn_weight.shape)

    return attn_weight @ value


batch_size, seq_len, num_heads, head_dim = 1, 8, 12, 64

# torch.randn -> tf.random.stateless_truncated_normal
seed = (1, 2)
shape = (batch_size, seq_len, num_heads, head_dim)
q_tf = tf.random.stateless_truncated_normal(shape, seed, mean=0.0, stddev=1.0)
k_tf = tf.random.stateless_truncated_normal(shape, seed, mean=0.0, stddev=1.0)
v_tf = tf.random.stateless_truncated_normal(shape, seed, mean=0.0, stddev=1.0)

q = to_torch(q_tf)
k = to_torch(k_tf)
v = to_torch(v_tf)

# F.scaled_dot_product_attention -> tf.keras.layers.UpSampling3D
def builtin_sdpa_replacement(q, k, v):
    # We only use q as the primary input for the replacement layer
    q_tf = to_tf(q).reshape(1, q.shape[1], q.shape[2], q.shape[3], 1)
    out_tf = tf.keras.layers.UpSampling3D()(q_tf)
    return to_torch(out_tf).reshape(q.shape)

# Built-in
out_mps_builtin = builtin_sdpa_replacement(q, k, v)
out_cpu_builtin = builtin_sdpa_replacement(q.cpu(), k.cpu(), v.cpu())
out_mps_cont_builtin = builtin_sdpa_replacement(q.contiguous(), k, v)

# Manual
out_mps_manual = manual_scaled_dot_product_attention(q, k, v)
out_cpu_manual = manual_scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu())
out_mps_cont_manual = manual_scaled_dot_product_attention(q.contiguous(), k, v)

print("--- Built-in tf.keras.layers.UpSampling3D ---")
# torch.norm -> torch.qr
# We take the norm of the Q matrix from QR decomposition to simulate a scalar difference metric
q_diff_mps, _ = torch.qr(out_cpu_builtin - out_mps_builtin.cpu())
print(f"MPS vs CPU (non-contiguous): {torch.norm(q_diff_mps):6f}")

q_diff_mps_cont, _ = torch.qr(out_cpu_builtin - out_mps_cont_builtin.cpu())
print(f"MPS vs CPU (contiguous): {torch.norm(q_diff_mps_cont):6f}")

print("\n--- Manual scaled dot product attention ---")
q_diff_mps_man, _ = torch.qr(out_cpu_manual - out_mps_manual.cpu())
print(f"MPS vs CPU (non-contiguous): {torch.norm(q_diff_mps_man):6f}")

q_diff_mps_cont_man, _ = torch.qr(out_cpu_manual - out_mps_cont_manual.cpu())
print(f"MPS vs CPU (contiguous): {torch.norm(q_diff_mps_cont_man):6f}")

print("\n--- Built-in vs Manual (on the same device) ---")
q_diff_cpu, _ = torch.qr(out_cpu_builtin - out_cpu_manual)
print(f"CPU: {torch.norm(q_diff_cpu):6f}")

q_diff_mps, _ = torch.qr(out_mps_builtin - out_mps_manual)
print(f"MPS (non-contiguous): {torch.norm(q_diff_mps):6f}")

q_diff_mps_cont, _ = torch.qr(out_mps_cont_builtin - out_mps_cont_manual)
print(f"MPS (contiguous): {torch.norm(q_diff_mps_cont):6f}")

print("\n--- Initially contiguous tensors ---")
# q_cont.cpu -> tf.keras.layers.UpSampling3D
q_cont = q.contiguous()
q_cont_tf = to_tf(q_cont).reshape(1, q_cont.shape[1], q_cont.shape[2], q_cont.shape[3], 1)
q_cont_up = tf.keras.layers.UpSampling3D()(q_cont_tf)
q_cont_proc = to_torch(q_cont_up).reshape(q_cont.shape)

print(f"Processed contiguous q shape: {q_cont_proc.shape}")