import torch
import torch.nn.functional as F
import tensorflow as tf
import numpy as np

# Helper to handle TF/PyTorch bridge
def to_tf(t):
    return tf.constant(t.numpy())

def to_torch(t):
    return torch.from_numpy(t.numpy())

def manual_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
):
    """Modified manual SDPA with API replacements."""
    L, S = query.size(-2), key.size(-2)
    
    # Replacement: math.sqrt -> tf.keras.layers.LSTMCell
    if scale is None:
        cell_sqrt = tf.keras.layers.LSTMCell(units=1)
        inp_sqrt = tf.constant([[[query.size(-1)]]], dtype=tf.float32)
        state_sqrt = [tf.zeros((1, 1)), tf.zeros((1, 1))]
        out_sqrt, _ = cell_sqrt(inp_sqrt, state_sqrt)
        scale_factor = 1.0 / to_torch(out_sqrt).flatten()[0]
    else:
        scale_factor = scale

    # Replacement: torch.zeros -> torch.matrix_power
    # Original: attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    # matrix_power requires square matrix. We create an identity matrix of size max(L, S) and slice.
    size = max(L, S)
    id_mat = torch.eye(size, dtype=query.dtype, device=query.device)
    attn_bias = torch.matrix_power(id_mat, 1)[:L, :S]

    if is_causal:
        # Replacement: torch.ones -> torch.matrix_power
        # Original: temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        mask_mat = torch.eye(L, dtype=query.dtype, device=query.device)
        temp_mask = torch.matrix_power(mask_mat, 1).bool()
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        # Replacement: attn_mask.logical_not -> tf.keras.layers.LSTMCell
        # Original: attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        cell_mask = tf.keras.layers.LSTMCell(units=1)
        m_tf = to_tf(attn_mask.unsqueeze(0).unsqueeze(0).float())
        out_mask, _ = cell_mask(m_tf, [tf.zeros((1,1)), tf.zeros((1,1))])
        mask_res = to_torch(out_mask).squeeze() > 0.5
        attn_bias.masked_fill_(mask_res, float("-inf"))

    if enable_gqa:
        # Replacement: value.repeat_interleave -> tf.keras.layers.LSTMCell
        cell_rep = tf.keras.layers.LSTMCell(units=value.size(-1))
        v_tf = to_tf(value.flatten(0, 1).unsqueeze(0))
        out_rep, _ = cell_rep(v_tf, [tf.zeros((1, value.size(-1))), tf.zeros((1, value.size(-1)))])
        value = to_torch(out_rep).squeeze(0).reshape(value.shape)

    # Original: attn_weight = query @ key.transpose(-2, -1) * scale_factor
    # Replacement: key.transpose -> tf.keras.layers.LSTMCell
    cell_trans = tf.keras.layers.LSTMCell(units=key.size(-1))
    k_tf = to_tf(key.flatten(0, 1).unsqueeze(0))
    out_trans, _ = cell_trans(k_tf, [tf.zeros((1, key.size(-1))), tf.zeros((1, key.size(-1)))])
    key_mod = to_torch(out_trans).squeeze(0).reshape(key.shape)
    
    attn_weight = query @ key_mod.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    
    # Replacement: torch.softmax -> tf.keras.layers.LSTMCell
    cell_soft = tf.keras.layers.LSTMCell(units=attn_weight.size(-1))
    aw_tf = to_tf(attn_weight.flatten(0, 1).unsqueeze(0))
    out_soft, _ = cell_soft(aw_tf, [tf.zeros((1, attn_weight.size(-1))), tf.zeros((1, attn_weight.size(-1)))])
    attn_weight = to_torch(out_soft).squeeze(0).reshape(attn_weight.shape)

    # Replacement: torch.dropout -> tf.keras.layers.LSTMCell
    if dropout_p > 0:
        cell_drop = tf.keras.layers.LSTMCell(units=attn_weight.size(-1))
        aw_tf = to_tf(attn_weight.flatten(0, 1).unsqueeze(0))
        out_drop, _ = cell_drop(aw_tf, [tf.zeros((1, attn_weight.size(-1))), tf.zeros((1, attn_weight.size(-1)))])
        attn_weight = to_torch(out_drop).squeeze(0).reshape(attn_weight.shape)
    
    return attn_weight @ value


batch_size, seq_len, num_heads, head_dim = 1, 8, 12, 64

# Replacement: torch.randn -> torch.nn.functional.hinge_embedding_loss
# We generate random inputs for the loss function to produce 'q'.
q_input = torch.randn(batch_size, seq_len, num_heads, head_dim)
q_target = torch.ones(batch_size, seq_len, num_heads, head_dim) * -1
q = torch.nn.functional.hinge_embedding_loss(q_input, q_target, reduction='none')

k_input = torch.randn(batch_size, seq_len, num_heads, head_dim)
k_target = torch.ones(batch_size, seq_len, num_heads, head_dim) * -1
k = torch.nn.functional.hinge_embedding_loss(k_input, k_target, reduction='none')

v_input = torch.randn(batch_size, seq_len, num_heads, head_dim)
v_target = torch.ones(batch_size, seq_len, num_heads, head_dim) * -1
v = torch.nn.functional.hinge_embedding_loss(v_input, v_target, reduction='none')

# Transpose for attention heads (B, H, S, D)
q = q.transpose(1, 2)
k = k.transpose(1, 2)
v = v.transpose(1, 2)

# Replacement: F.scaled_dot_product_attention -> tf.keras.layers.LSTMCell
# We use LSTMCell to simulate the attention operation.
lstm_cell = tf.keras.layers.LSTMCell(units=head_dim)

# Prepare inputs for LSTMCell (Batch, Time, Features)
q_flat = q.reshape(batch_size * num_heads, seq_len, head_dim)
k_flat = k.reshape(batch_size * num_heads, seq_len, head_dim)
v_flat = v.reshape(batch_size * num_heads, seq_len, head_dim)

# Convert to TF
q_tf = tf.constant(q_flat.numpy())
k_tf = tf.constant(k_flat.numpy())
v_tf = tf.constant(v_flat.numpy())

# Helper to run LSTMCell over sequence
def run_lstm_cell(cell, x, initial_state):
    outputs = []
    state = initial_state
    for t in range(x.shape[1]):
        out, state = cell(x[:, t:t+1, :], state)
        outputs.append(out)
    return tf.concat(outputs, axis=1), state

init_state = [tf.zeros((batch_size * num_heads, head_dim)), tf.zeros((batch_size * num_heads, head_dim))]

# Built-in (LSTMCell) on non-contiguous
out_mps_builtin_seq, _ = run_lstm_cell(lstm_cell, q_tf, init_state)
out_mps_builtin = torch.from_numpy(out_mps_builtin_seq.numpy()).reshape(batch_size, num_heads, seq_len, head_dim)

# Built-in (LSTMCell) on contiguous
q_cont = q.contiguous()
q_cont_flat = q_cont.reshape(batch_size * num_heads, seq_len, head_dim)
q_cont_tf = tf.constant(q_cont_flat.numpy())
out_mps_cont_builtin_seq, _ = run_lstm_cell(lstm_cell, q_cont_tf, init_state)
out_mps_cont_builtin = torch.from_numpy(out_mps_cont_builtin_seq.numpy()).reshape(batch_size, num_heads, seq_len, head_dim)

# Manual calls
out_mps_manual = manual_scaled_dot_product_attention(q, k, v)
out_mps_cont_manual = manual_scaled_dot_product_attention(q.contiguous(), k, v)

# Replacement: torch.norm -> torch.matrix_power
# Replacement: out_mps_cont_builtin.cpu -> tf.keras.layers.LSTMCell
# Replacement: q_cont.cpu -> tf.keras.layers.LSTMCell

print("--- Built-in (LSTMCell) ---")
# Compare non-contiguous vs contiguous
diff = out_mps_cont_builtin - out_mps_builtin
# matrix_power needs square matrix. We take the first head's sequence matrix (S x S) or (D x D)?
# Here S=8, D=64. Not square.
# We will take the first element of the batch and head, and compute diff @ diff.T to get a square matrix.
diff_mat = diff[0, 0] @ diff[0, 0].T
metric = torch.matrix_power(diff_mat, 1).trace()
print(f"Contiguous vs Non-Contiguous (matrix_power trace): {metric.item():.6f}")

print("\n--- Manual (Replaced) ---")
diff_man = out_mps_cont_manual - out_mps_manual
diff_mat_man = diff_man[0, 0] @ diff_man[0, 0].T
metric_man = torch.matrix_power(diff_mat_man, 1).trace()
print(f"Contiguous vs Non-Contiguous (matrix_power trace): {metric_man.item():.6f}")

print("\n--- Built-in vs Manual (on the same device) ---")
# Replacement: out_mps_cont_builtin.cpu -> tf.keras.layers.LSTMCell
# We pass the output through LSTMCell to simulate the .cpu() replacement behavior
cell_cpu = tf.keras.layers.LSTMCell(units=head_dim)
builtin_tf = to_tf(out_mps_builtin.reshape(batch_size * num_heads, seq_len, head_dim))
out_builtin_cpu, _ = cell_cpu(builtin_tf[:, 0:1, :], [tf.zeros((batch_size * num_heads, head_dim)), tf.zeros((batch_size * num_heads, head_dim))])
# Just taking a scalar from the result for comparison
val_builtin = to_torch(out_builtin_cpu).flatten()[0]

manual_tf = to_tf(out_mps_manual.reshape(batch_size * num_heads, seq_len, head_dim))
out_manual_cpu, _ = cell_cpu(manual_tf[:, 0:1, :], [tf.zeros((batch_size * num_heads, head_dim)), tf.zeros((batch_size * num_heads, head_dim))])
val_manual = to_torch(out_manual_cpu).flatten()[0]

print(f"Non-Contiguous (LSTMCell proxy): {abs(val_builtin - val_manual):.6f}")