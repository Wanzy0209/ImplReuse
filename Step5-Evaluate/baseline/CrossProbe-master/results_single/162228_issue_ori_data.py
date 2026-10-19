# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
from torch.nn.attention.flex_attention import flex_attention

@torch.compile()
def test(x, y):
    # Materialize a bias matrix
    B, L, device = x.shape[0], x.shape[1], x.device
    b = torch.arange(B, device=device, dtype=torch.long).view(B, 1, 1)
    q_idx = torch.arange(L, device=device, dtype=torch.long).view(1, L, 1)
    kv_idx = torch.arange(L, device=device, dtype=torch.long).view(1, 1, L)
    bias_mat = y[b, q_idx] + y[b, kv_idx] # (B, L, L)

    # Dummy score_mod retrieving bias values
    def score_mod(score, b, h, q_idx, kv_idx):
        return score + bias_mat[b, q_idx, kv_idx]

    x_ = x[:, :, None].repeat(1, 1, 16, 1)
    # torch._dynamo.graph_break()
    return flex_attention(x_, x_, x_, score_mod=score_mod)


DEVICE = "cuda"
B, L, D = 2, 16, 64

x = torch.randn(B, L, D, device=DEVICE, requires_grad=True)
y = torch.randn(B, L, device=DEVICE, requires_grad=True)

out = test(x, y).mean().backward()

print(torch.__version__)
print(f"x: {(x.grad is not None) and (x.grad.norm() > 0)}, y: {(y.grad is not None) and (y.grad.norm() > 0)}")
assert x.grad.norm() > 0
assert y.grad.norm() > 0