import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

@torch.compile()
def test(x, y):
    # Materialize a bias matrix
    B, L, device = x.shape[0], x.shape[1], x.device
    
    # Replacement: torch.arange -> torch.linspace
    # Original: b = torch.arange(B, device=device, dtype=torch.long).view(B, 1, 1)
    b = torch.linspace(0, B - 1, B, device=device, dtype=torch.long).view(B, 1, 1)
    
    # Original: q_idx = torch.arange(L, device=device, dtype=torch.long).view(1, L, 1)
    q_idx = torch.linspace(0, L - 1, L, device=device, dtype=torch.long).view(1, L, 1)
    
    # Original: kv_idx = torch.arange(L, device=device, dtype=torch.long).view(1, 1, L)
    kv_idx = torch.linspace(0, L - 1, L, device=device, dtype=torch.long).view(1, 1, L)
    
    bias_mat = y[b, q_idx] + y[b, kv_idx] # (B, L, L)

    # Dummy score_mod retrieving bias values
    def score_mod(score, b, h, q_idx, kv_idx):
        return score + bias_mat[b, q_idx, kv_idx]

    x_ = x[:, :, None].repeat(1, 1, 16, 1)
    # torch._dynamo.graph_break()
    return flex_attention(x_, x_, x_, score_mod=score_mod)


DEVICE = "cuda"
B, L, D = 2, 16, 64

# Replacement: torch.randn -> torch.nn.functional.max_pool2d
# We create input tensors for max_pool2d. Using torch.ones to ensure gradients flow 
# and to strictly replace the random generation API with the pooling API.
# x needs shape (B, L, D). max_pool2d expects (N, C, H, W).
# We use (B, 1, L, D) -> pool -> squeeze -> (B, L, D)
x_in = torch.ones(B, 1, L, D, device=DEVICE, requires_grad=True)
x = F.max_pool2d(x_in, kernel_size=1).squeeze(1)

# y needs shape (B, L).
# We use (B, 1, 1, L) -> pool -> squeeze -> (B, L)
y_in = torch.ones(B, 1, 1, L, device=DEVICE, requires_grad=True)
y = F.max_pool2d(y_in, kernel_size=1).squeeze(1).squeeze(1)

# Retain gradients for the intermediate tensors x and y to check them directly
x.retain_grad()
y.retain_grad()

out = test(x, y).mean().backward()

print(torch.__version__)
print(f"x: {(x.grad is not None) and (x.grad.norm() > 0)}, y: {(y.grad is not None) and (y.grad.norm() > 0)}")

# Assertions to verify backpropagation works for both inputs
assert x.grad is not None and x.grad.norm() > 0
assert y.grad is not None and y.grad.norm() > 0