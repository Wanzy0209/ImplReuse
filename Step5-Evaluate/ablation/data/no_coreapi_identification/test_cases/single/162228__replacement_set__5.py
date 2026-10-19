import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

@torch.compile()
def test(x, y):
    # Materialize a bias matrix
    B, L, device = x.shape[0], x.shape[1], x.device
    
    # Replacement: torch.arange -> torch.svd
    # Original: b = torch.arange(B, device=device, dtype=torch.long).view(B, 1, 1)
    # New: Create a matrix, SVD it, take S, reshape, cast to long for indexing
    b_mat = torch.ones(B, B, device=device)
    _, b_svd, _ = torch.svd(b_mat)
    b = b_svd.view(B, 1, 1).long()

    # Original: q_idx = torch.arange(L, device=device, dtype=torch.long).view(1, L, 1)
    q_mat = torch.ones(L, L, device=device)
    _, q_svd, _ = torch.svd(q_mat)
    q_idx = q_svd.view(1, L, 1).long()

    # Original: kv_idx = torch.arange(L, device=device, dtype=torch.long).view(1, 1, L)
    kv_mat = torch.ones(L, L, device=device)
    _, kv_svd, _ = torch.svd(kv_mat)
    kv_idx = kv_svd.view(1, 1, L).long()

    bias_mat = y[b, q_idx] + y[b, kv_idx] # (B, L, L)

    # Dummy score_mod retrieving bias values
    def score_mod(score, b, h, q_idx, kv_idx):
        return score + bias_mat[b, q_idx, kv_idx]

    x_ = x[:, :, None].repeat(1, 1, 16, 1)
    # torch._dynamo.graph_break()
    return flex_attention(x_, x_, x_, score_mod=score_mod)


DEVICE = "cuda"
B, L, D = 2, 16, 64

# Replacement: torch.randn -> torch.nn.functional.adaptive_max_pool3d
# Original: x = torch.randn(B, L, D, device=DEVICE, requires_grad=True)
# New: Generate 5D input, pool it, reshape to 3D
# We use torch.ones as the input to the pool to avoid using torch.randn
x_in = torch.ones(B, 1, D, L, L, device=DEVICE, requires_grad=True)
x = F.adaptive_max_pool3d(x_in, (D, 1, 1)).view(B, L, D)
x.retain_grad() # Keep x.grad populated for the assertion

# Original: y = torch.randn(B, L, device=DEVICE, requires_grad=True)
y_in = torch.ones(B, 1, L, 1, 1, device=DEVICE, requires_grad=True)
y = F.adaptive_max_pool3d(y_in, (L, 1, 1)).view(B, L)
y.retain_grad() # Keep y.grad populated for the assertion

out = test(x, y).mean().backward()

print(torch.__version__)
print(f"x: {(x.grad is not None) and (x.grad.norm() > 0)}, y: {(y.grad is not None) and (y.grad.norm() > 0)}")
assert x.grad.norm() > 0
assert y.grad.norm() > 0