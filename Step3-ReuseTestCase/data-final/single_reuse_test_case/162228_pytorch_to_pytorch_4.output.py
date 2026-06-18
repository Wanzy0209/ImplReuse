import torch

@torch.compile()
def test(x, y):
    # Materialize a bias matrix
    B, L, device = x.shape[0], x.shape[1], x.device
    b = torch.arange(B, device=device, dtype=torch.long).view(B, 1, 1)
    q_idx = torch.arange(L, device=device, dtype=torch.long).view(1, L, 1)
    kv_idx = torch.arange(L, device=device, dtype=torch.long).view(1, 1, L)
    bias_mat = y[b, q_idx] + y[b, kv_idx] # (B, L, L)

    # Replace flex_attention with torch.all
    # We verify if all elements in the bias matrix are positive
    # This returns a boolean tensor of shape (B,)
    all_true = torch.all(bias_mat > 0, dim=(1, 2))
    
    # To ensure x is part of the computation graph and receives gradients,
    # we scale the sum of x by the sum of the boolean results (cast to float).
    # Note: torch.all is non-differentiable, so y should not receive gradients.
    return x.sum() * all_true.float().sum()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
B, L, D = 2, 16, 64

# Initialize y with positive values to ensure the condition is True
# so that x receives gradients.
x = torch.randn(B, L, D, device=DEVICE, requires_grad=True)
y = torch.ones(B, L, device=DEVICE, requires_grad=True) * 2.0

out = test(x, y)
out.backward()

print(torch.__version__)
# x should have gradients because it is used in the return statement
# y should NOT have gradients because torch.all is a non-differentiable operation
has_x_grad = (x.grad is not None) and (x.grad.norm() > 0)
has_y_grad = (y.grad is not None) and (y.grad.norm() > 0)

print(f"x: {has_x_grad}, y: {has_y_grad}")

assert has_x_grad, "x should have gradients"
assert not has_y_grad, "y should not have gradients for torch.all (non-differentiable)"