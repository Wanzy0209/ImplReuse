import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

@torch.compile()
def test(x, y):
    # Materialize a bias matrix
    B, L, device = x.shape[0], x.shape[1], x.device
    
    # Replacement: torch.arange -> torch.quantile
    # We use torch.linspace to generate a sequence and torch.quantile to select values, 
    # mimicking the index generation of torch.arange.
    b_src = torch.linspace(0, B - 1, B, device=device).view(B, 1)
    b = torch.quantile(b_src, torch.linspace(0, 1, B, device=device), dim=0).long().view(B, 1, 1)
    
    q_src = torch.linspace(0, L - 1, L, device=device).view(1, L)
    q_idx = torch.quantile(q_src, torch.linspace(0, 1, L, device=device), dim=1).long().view(1, L, 1)
    
    kv_src = torch.linspace(0, L - 1, L, device=device).view(1, L)
    kv_idx = torch.quantile(kv_src, torch.linspace(0, 1, L, device=device), dim=1).long().view(1, 1, L)

    bias_mat = y[b, q_idx] + y[b, kv_idx] # (B, L, L)

    # Dummy score_mod retrieving bias values
    def score_mod(score, b, h, q_idx, kv_idx):
        return score + bias_mat[b, q_idx, kv_idx]

    x_ = x[:, :, None].repeat(1, 1, 16, 1)
    # torch._dynamo.graph_break()
    return flex_attention(x_, x_, x_, score_mod=score_mod)


DEVICE = "cuda"
B, L, D = 2, 16, 64

# Replacement: torch.randn -> torch.nn.functional.adaptive_max_pool2d
# We generate random 4D tensors, apply adaptive max pooling, and then detach/require_grad 
# to maintain the leaf variable status expected by the test assertions.
x_src = torch.randn(B, 1, L, D, device=DEVICE)
x = F.adaptive_max_pool2d(x_src, (L, D)).squeeze(1).detach().requires_grad_(True)

y_src = torch.randn(B, 1, 1, L, device=DEVICE)
y = F.adaptive_max_pool2d(y_src, (1, L)).squeeze(1).squeeze(1).detach().requires_grad_(True)

out = test(x, y).mean().backward()

print(torch.__version__)
print(f"x: {(x.grad is not None) and (x.grad.norm() > 0)}, y: {(y.grad is not None) and (y.grad.norm() > 0)}")
assert x.grad.norm() > 0
assert y.grad.norm() > 0