import torch

@torch.compile()
def test(x, y):
    # Materialize a bias matrix
    B, L, device = x.shape[0], x.shape[1], x.device
    b = torch.arange(B, device=device, dtype=torch.long).view(B, 1, 1)
    q_idx = torch.arange(L, device=device, dtype=torch.long).view(1, L, 1)
    kv_idx = torch.arange(L, device=device, dtype=torch.long).view(1, 1, L)
    bias_mat = y[b, q_idx] + y[b, kv_idx] # (B, L, L)

    # Adaptation: Use torch.prod instead of flex_attention
    # We combine x and bias_mat to ensure gradients flow to both inputs
    x_ = x[:, :, None].repeat(1, 1, 16, 1) # (B, L, 16, D)
    
    # Force graph break to test behavior similar to the original bug report
    torch._dynamo.graph_break()

    # Reshape bias_mat to broadcast with x_
    # bias_mat is (B, L, L), sum over last dim to get (B, L, 1, 1)
    bias_reshaped = bias_mat.sum(dim=-1, keepdim=True).unsqueeze(-1)
    
    combined = x_ + bias_reshaped
    return torch.prod(combined)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
B, L, D = 2, 16, 64

x = torch.randn(B, L, D, device=DEVICE, requires_grad=True)
y = torch.randn(B, L, device=DEVICE, requires_grad=True)

out = test(x, y)
# torch.prod returns a scalar, so we can call backward directly
out.backward()

print(torch.__version__)
print(f"x: {(x.grad is not None) and (x.grad.norm() > 0)}, y: {(y.grad is not None) and (y.grad.norm() > 0)}")
assert x.grad is not None and x.grad.norm() > 0
assert y.grad is not None and y.grad.norm() > 0