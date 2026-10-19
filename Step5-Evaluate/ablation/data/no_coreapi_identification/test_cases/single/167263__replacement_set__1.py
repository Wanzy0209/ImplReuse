import torch
import torch.nn as nn

B, H, W, C = 20, 2, 2, 128

# Replacement: torch.randn -> torch.var_mean
# We use var_mean on an empty tensor, take the variance (scalar), and expand it to the required shape.
# This preserves the shape (B, H, W, C) needed for the test logic.
x = torch.var_mean(torch.empty(B, H, W, C))[0].expand(B, H, W, C).detach().requires_grad_(True)

linear = nn.Linear(C, C, bias=False)

values = linear(x)
values.retain_grad()

values_view = values.view(B, H * W, C)
values_view.retain_grad()

# Capture stride for assertion
grad_stride = None

def log_grad(name):
    def hook(grad):
        nonlocal grad_stride
        print(f"{name} hook - shape: {grad.shape}, stride: {grad.stride()}")
        grad_stride = grad.stride()
        return grad
    return hook

print("forward strides", values_view.shape, values_view.stride())
values_view.register_hook(log_grad("values_view"))

# Replacement: torch.randn -> torch.var_mean
weights = torch.var_mean(torch.empty(B, H * W, C))[0].expand(B, H * W, C)

result = torch.einsum("bhc,bhc->bc", weights, values_view)

# Replacement: torch.ones_like -> torch.empty_like
result.backward(torch.empty_like(result))

# Assertion to verify the fix: gradient stride should match input stride
# The bug report indicates that the strides differ, causing failures.
assert grad_stride == values_view.stride(), f"Gradient stride {grad_stride} does not match input stride {values_view.stride()}"