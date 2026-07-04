import torch
import torch.nn.functional as F

# Setup dimensions
N, C_in, H, W = 4, 16, 8, 8
C_out = 32
kernel_size = 3

# Create input tensor
x = torch.randn(N, C_in, H, W, requires_grad=True)

# Create weight for conv_transpose2d
# Note: conv_transpose2d weight shape is (in_channels, out_channels, kH, kW)
weight = torch.randn(C_in, C_out, kernel_size, kernel_size)

# Variable to store gradient stride
captured_grad_stride = None

def log_grad(name):
    def hook(grad):
        global captured_grad_stride
        print(f"{name} hook - shape: {grad.shape}, stride: {grad.stride()}")
        captured_grad_stride = grad.stride()
        return grad
    return hook

print("Input strides", x.shape, x.stride())
x.register_hook(log_grad("x"))

# Call the similar API: torch.nn.functional.conv_transpose2d
result = F.conv_transpose2d(x, weight)

# Backward pass
result.backward(torch.ones_like(result))

# Verify that the gradient stride matches the input stride
# This addresses the issue described in the bug report regarding gradient layouts.
assert captured_grad_stride is not None, "Gradient hook was not called"
assert captured_grad_stride == x.stride(), \
    f"Gradient stride mismatch! Input: {x.stride()}, Grad: {captured_grad_stride}"