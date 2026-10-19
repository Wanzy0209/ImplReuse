import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

@torch.compile()
def test(x, y):
    # Materialize a bias matrix
    B, L, device = x.shape[0], x.shape[1], x.device
    
    # Replacement: torch.arange -> torch.norm
    # Original: b = torch.arange(B, device=device, dtype=torch.long).view(B, 1, 1)
    # New: Use torch.norm on a tensor of ones to generate values, then cast to long for indexing
    b = torch.norm(torch.ones(B, 1, 1, device=device), dim=[], keepdim=True).long()
    
    # Original: q_idx = torch.arange(L, device=device, dtype=torch.long).view(1, L, 1)
    q_idx = torch.norm(torch.ones(1, L, 1, device=device), dim=[], keepdim=True).long()
    
    # Original: kv_idx = torch.arange(L, device=device, dtype=torch.long).view(1, 1, L)
    kv_idx = torch.norm(torch.ones(1, 1, L, device=device), dim=[], keepdim=True).long()

    bias_mat = y[b, q_idx] + y[b, kv_idx] # (B, L, L)

    # Dummy score_mod retrieving bias values
    def score_mod(score, b, h, q_idx, kv_idx):
        return score + bias_mat[b, q_idx, kv_idx]

    x_ = x[:, :, None].repeat(1, 1, 16, 1)
    # torch._dynamo.graph_break()
    return flex_attention(x_, x_, x_, score_mod=score_mod)


DEVICE = "cuda"
B, L, D = 2, 16, 64

# Replacement: torch.randn -> torch.nn.functional.max_pool3d
# Original: x = torch.randn(B, L, D, device=DEVICE, requires_grad=True)
# New: Create a 5D tensor, apply max_pool3d with kernel_size=1 (identity-like), and squeeze to 3D
raw_x = torch.ones(B, 1, 1, L, D, device=DEVICE, requires_grad=True)
x = F.max_pool3d(raw_x, kernel_size=1).squeeze().requires_grad_(True)

# Original: y = torch.randn(B, L, device=DEVICE, requires_grad=True)
raw_y = torch.ones(B, 1, 1, L, 1, device=DEVICE, requires_grad=True)
y = F.max_pool3d(raw_y, kernel_size=1).squeeze().requires_grad_(True)

out = test(x, y).mean().backward()

print(torch.__version__)
print(f"x: {(x.grad is not None) and (x.grad.norm() > 0)}, y: {(y.grad is not None) and (y.grad.norm() > 0)}")
assert x.grad is not None
assert x.grad.norm() > 0
assert y.grad is not None
assert y.grad.norm() > 0