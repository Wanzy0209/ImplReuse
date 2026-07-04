import torch
import torch.nn as nn

B, H, W, C = 20, 2, 2, 128

x = torch.randn(B, H, W, C, requires_grad=True)
linear = nn.Linear(C, C, bias=False)

values = linear(x)
values_view = values.view(B, H * W, C)
values_view.retain_grad()


def log_grad(name):
    def hook(grad):
        print(f"{name} hook - shape: {grad.shape}, stride: {grad.stride()}")
        return grad

    return hook


print("forward strides", values_view.shape, values_view.stride())
values_view.register_hook(log_grad("values_view"))

weights = torch.randn(B, H * W, C)

# Replaced torch.floor_divide with torch.div
# floor_divide is not differentiable. torch.div (true division) is differentiable
# and serves the same purpose of testing stride propagation through an element-wise op.
result = torch.div(weights, values_view)
result = result.sum()

result.backward(torch.ones_like(result))

# Verify that the gradient stride matches the input stride
# This assertion checks if the similar API (div) respects the stride conventions
# that were reported as problematic in the original API (einsum).
assert values_view.stride() == values_view.grad.stride(), \
    f"Gradient stride mismatch: Input stride {values_view.stride()}, Grad stride {values_view.grad.stride()}"