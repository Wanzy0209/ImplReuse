import torch
import torch.nn as nn

B, H, W, C = 20, 2, 2, 128

x = torch.randn(B, H, W, C, requires_grad=True)
linear = nn.Linear(C, C, bias=False)

values = linear(x)
values_view = values.view(B, H * W, C)

# Variable to store the gradient stride captured in the hook
captured_grad_stride = None

def hook(grad):
    global captured_grad_stride
    captured_grad_stride = grad.stride()
    return grad

print("Forward strides - shape:", values_view.shape, "stride:", values_view.stride())
values_view.register_hook(hook)

# Fix: Call retain_grad() on the non-leaf tensor to populate its .grad attribute during backward()
values_view.retain_grad()

# Replace torch.einsum with torch.nn.functional.softshrink
result = torch.nn.functional.softshrink(values_view)

result.backward(torch.ones_like(result))

print("Gradient strides - shape:", values_view.grad.shape, "stride:", captured_grad_stride)

# Assert that the gradient stride matches the input stride
assert values_view.stride() == captured_grad_stride, \
    f"Stride mismatch: forward stride is {values_view.stride()}, but gradient stride is {captured_grad_stride}"