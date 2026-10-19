import torch
import tensorflow as tf
from torch.nn.attention.flex_attention import flex_attention

@torch.compile()
def test(x, y):
    # Materialize a bias matrix
    B, L, device = x.shape[0], x.shape[1], x.device
    
    # Replacement: torch.arange -> torch.std
    # We use torch.std on a dummy tensor with an extra dimension to generate a tensor of the desired shape.
    # Since std of ones is 0, this creates index tensors filled with 0s.
    b = torch.std(torch.ones(B, 1, 1, 1, device=device), dim=-1).long()
    q_idx = torch.std(torch.ones(1, L, 1, 1, device=device), dim=-1).long()
    kv_idx = torch.std(torch.ones(1, 1, L, 1, device=device), dim=-1).long()
    
    bias_mat = y[b, q_idx] + y[b, kv_idx] # (B, L, L)

    # Dummy score_mod retrieving bias values
    def score_mod(score, b, h, q_idx, kv_idx):
        return score + bias_mat[b, q_idx, kv_idx]

    x_ = x[:, :, None].repeat(1, 1, 16, 1)
    # torch._dynamo.graph_break()
    return flex_attention(x_, x_, x_, score_mod=score_mod)


DEVICE = "cuda"
B, L, D = 2, 16, 64

# Replacement: torch.randn -> tf.random.stateless_truncated_normal
# Generate tensors using TensorFlow and convert to PyTorch
x_tf = tf.random.stateless_truncated_normal(shape=(B, L, D), seed=[0, 1])
y_tf = tf.random.stateless_truncated_normal(shape=(B, L), seed=[1, 2])

x = torch.from_numpy(x_tf.numpy()).to(DEVICE).requires_grad_(True)
y = torch.from_numpy(y_tf.numpy()).to(DEVICE).requires_grad_(True)

out = test(x, y).mean().backward()

print(torch.__version__)
print(f"x: {(x.grad is not None) and (x.grad.norm() > 0)}, y: {(y.grad is not None) and (y.grad.norm() > 0)}")
assert x.grad.norm() > 0
assert y.grad.norm() > 0