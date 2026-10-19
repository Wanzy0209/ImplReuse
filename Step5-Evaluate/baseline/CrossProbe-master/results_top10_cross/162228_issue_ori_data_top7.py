```python
import tensorflow as tf

# PyTorch's flex_attention is a specific API for flexible attention mechanisms.
# TensorFlow does not have a direct 1:1 equivalent, so we implement a standard
# scaled dot-product attention mechanism that supports the score_mod callback.
def flex_attention_tf(q, k, v, score_mod):
    # q, k, v shape: (B, L, H, D)
    B, L, H, D = q.shape

    # Transpose for matmul: (B, H, L, D)
    q_t = tf.transpose(q, [0, 2, 1, 3])
    k_t = tf.transpose(k, [0, 2, 1, 3])
    v_t = tf.transpose(v, [0, 2, 1, 3])

    # Scaled Dot-Product Attention
    # scores: (B, H, L, L)
    scores = tf.matmul(q_t, k_t, transpose_b=True) / tf.math.sqrt(tf.cast(D, tf.float32))

    # Create indices for score_mod
    # b_idx: (B, 1, 1, 1) -> broadcast to (B, H, L, L)
    # q_idx: (1, 1, L, 1) -> broadcast to (B, H, L, L)
    # kv_idx: (1, 1, 1, L) -> broadcast to (B, H, L, L)
    b_idx = tf.range(B, dtype=tf.int32)
    b_idx = tf.reshape(b_idx, (B, 1, 1, 1))

    q_idx = tf.range(L, dtype=tf.int32)
    q_idx = tf.reshape(q_idx, (1, 1, L, 1))

    kv_idx = tf.range(L, dtype=tf.int32)
    kv_idx = tf.reshape(kv_idx, (1, 1, 1, L))

    # Broadcast indices to match scores shape
    b_idx_b = tf.broadcast_to(b_idx, tf.shape(scores))
    q_idx_b = tf.broadcast_to(q_idx, tf.shape(scores))
    kv_idx_b = tf.broadcast_to(kv_idx, tf.shape(scores))

    # Apply score_mod
    # Note: The score_mod in the source captures bias_mat.
    # We pass the indices to it.
    modified_scores = score_mod(scores, b_idx_b, q_idx_b, kv_idx_b)

    # Softmax
    attn_weights = tf.nn.softmax(modified_scores, axis=-1)

    # Output
    output = tf.matmul(attn_weights, v_t) # (B, H, L, D)
    output = tf.transpose(output, [0, 2, 1, 3]) # (B, L, H, D)
    return output

# tf.function is the TensorFlow equivalent to torch.compile
@tf.function
def test(x, y):
    # Materialize a bias matrix
    B, L = x.shape[0], x.shape[1]
    # device is implicit in TensorFlow, usually handled by placement context
    
    # b = torch.arange(B, device=device, dtype=torch.long).view(B, 1, 1)
    b = tf.range(B, dtype=tf.int32)
    b = tf.reshape(b, (B, 1, 1))
    
    # q_idx = torch.arange(L, device=device, dtype=torch.long).view(1, L, 1)
    q_idx = tf.range(L, dtype=tf.int32)
    q_idx = tf.reshape(q_idx, (1, L, 1))
    
    # kv_idx = torch.arange(L, device=device, dtype=torch.long).view(1, 1, L)
    kv_idx = tf.range(L, dtype=tf.int32)
    kv_idx = tf.reshape(kv_idx, (1, 1, L))
    
    # bias_mat = y[b, q_idx] + y[b, kv_idx] # (B, L, L)
    # Implementing advanced indexing y[b, q_idx] using tf.gather_nd
    # y is (B, L). b is (B, 1, 1). q_idx is (1, L, 1).
    # We construct coordinates (B, L, 2) for gather_nd
    coords_b = tf.broadcast_to(b, [B, L, 1])
    coords_q = tf.broadcast_to(q_idx, [B, L, 1])
    coords = tf.concat([coords_b, coords_q], axis=-1)
    y_b_q = tf.gather_nd(y, coords) # (B, L)
    y_b_q = tf.expand_dims(y_b_q, axis=-1) # (B, L, 1)

    # Implementing y[b, kv_idx]
    # b is (B, 1, 1), kv_idx is (1, 1, L). Result (B, 1, L).
    coords_b_2 = tf.broadcast_to(b, [B, 1, L])
    coords_kv = tf.broadcast_to(kv_idx, [B, 1, L])
    coords_2 = tf.concat([coords_b_2, coords_kv], axis=-1)
    y_b_kv = tf.gather_nd(y, coords_2) # (B, 1, L)

    bias_mat = y_b_q + y_b_kv # (B, L, L)

    # Dummy score_mod retrieving bias values
    # PyTorch signature: (score, b, h, q_idx, kv_idx)
    def score_mod(score, b_idx, q_idx, kv_idx):
        # score shape: (B, H, L, L)
        # b_idx, q_idx, kv_idx are broadcasted to (B, H, L, L)
        # bias_mat is (B, L, L).
        # We need to gather from bias_mat using (b_idx, q_idx, kv_idx).
        # Since bias_mat doesn't have H dimension, we ignore H in indices.
        b_idx_2d = b_idx[:, 0, :, :] # (B, L, L)
        q_idx_2d = q_idx[:, 0, :, :] # (B, L, L)
        kv_idx_2d = kv_idx[:, 0, :, :] # (B, L, L)

        # Stack to form (B, L, L, 3)
        indices = tf.stack([b_idx_2d, q_idx_2d, kv_idx_2d], axis=-1)
        # Gather from bias_mat
        bias_values = tf.gather_nd(bias_mat, indices) # (B, L, L)
        # Expand bias_values to match score (B, H, L, L)
        bias_values = tf.expand_dims(bias_values, axis=1) # (B, 1, L, L)
        return score + bias_values

    # x_ = x[:, :, None].repeat(1, 1, 16, 1)
    # x is (B, L, D). x_ is (B, L, 16, D).
    x_ = tf.repeat(tf.expand_dims(x, axis=2), repeats=16, axis=2)
    
    # torch._dynamo.graph_break()
    # return flex_attention(x_, x_, x_, score_mod=score_mod)
    return flex_attention_tf(x_, x_, x_, score_mod=score_mod)


# DEVICE = "cuda" # TensorFlow handles device placement automatically
B, L, D = 2, 16, 64

# x = torch.randn(B, L, D, device=DEVICE, requires_grad=True)
# y = torch.randn(B, L, device=DEVICE, requires_grad=True)
x = tf.random.normal((B, L, D))
y = tf.random.normal((B, L))

# out = test(x, y).mean().backward()
# TensorFlow uses GradientTape for automatic differentiation
with tf.GradientTape() as tape:
    tape.watch(x)
    tape.watch(y)
    out = test(x, y)
    loss = tf.reduce_mean(out)

grads = tape.gradient(loss, [x, y])
x_grad = grads[0]
y_grad = grads[1]

# print(torch.__version__)
print(f"TensorFlow Version: {tf.__version__}")

# print(f"x: {(x.grad is not None) and (x.grad.norm() > 0)}, y: {(y.grad is not None) and (y.grad.norm() > 0)}")
# Check if gradients exist and norm > 0
x_valid = (x_grad is not None) and (tf.norm(x_grad) > 0)
y_valid = (y_grad is not None) and (tf.norm(y_grad) > 0)
print(f"x: {x_valid}, y: {y_valid}")

# assert x.grad.norm() > 0
# assert y.grad.norm() > 0
assert tf.norm(x_grad) > 0
assert tf.norm(y_grad) > 0
```