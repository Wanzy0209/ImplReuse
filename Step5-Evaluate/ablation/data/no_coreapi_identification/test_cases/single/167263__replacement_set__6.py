import torch
import torch.nn as nn
import torch.nn.functional as F

B, H, W, C = 20, 2, 2, 128

# Replacement: torch.randn -> torch.nn.functional.max_pool1d
# We create a dummy input to pass through max_pool1d to generate 'x'.
# Target shape for x is (B, H, W, C) = (20, 2, 2, 128).
# max_pool1d expects (N, C, L). We treat C as channels.
# We need the output length L to be H * W = 4.
# Using kernel_size=2, stride=2, we need input length 8.
dummy_input = torch.ones(B, C, H * W * 2, requires_grad=True)
x_pooled = F.max_pool1d(dummy_input, kernel_size=2, stride=2)
x = x_pooled.view(B, H, W, C)

linear = nn.Linear(C, C, bias=False)

values = linear(x)
values.retain_grad()

values_view = values.view(B, H * W, C)
values_view.retain_grad()

# Hook to capture gradient information
grad_info = {}

def log_grad(name):
    def hook(grad):
        grad_info[name] = {"shape": grad.shape, "stride": grad.stride()}
        return grad
    return hook

print("forward strides", values_view.shape, values_view.stride())
values_view.register_hook(log_grad("values_view"))

# Replacing torch.randn for weights with torch.ones to avoid the API call
weights = torch.ones(B, H * W, C)

result = torch.einsum("bhc,bhc->bc", weights, values_view)

# Replacement: torch.ones_like -> torch.minimum
# We use torch.minimum to generate the gradient tensor for the backward pass.
# torch.minimum(result, result) returns a tensor with the same values as result.
grad_input = torch.minimum(result, result)
result.backward(grad_input)

# Assertions to verify the test case ran and captured the gradient
assert "values_view" in grad_info, "Gradient hook was not triggered"
assert grad_info["values_view"]["shape"] == values_view.shape, "Gradient shape mismatch"

print("Backward strides", grad_info["values_view"]["shape"], grad_info["values_view"]["stride"])
print("Test passed.")