import torch
from torch.library import Library, impl

# Setup: Define a custom library and operator to test autograd registration
lib = Library("test_custom_lib", "DEF")
lib.define("custom_op(Tensor x) -> Tensor")

# Define the autograd function using torch.autograd.Function
# This is the standard way to define custom ops with backward support
class CustomOpFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x * 2.0

    @staticmethod
    def backward(ctx, grad_output):
        # Gradient of x * 2 w.r.t x is 2
        return grad_output * 2.0

# Register the implementation for the CPU backend
# We wrap the autograd function's apply method to enable autograd
@impl(lib, "custom_op", "CPU")
def custom_op_cpu(x):
    return CustomOpFunc.apply(x)

# Verification: Ensure the operator works correctly with autograd
x = torch.rand((1, 3, 224, 224), requires_grad=True)
y = torch.ops.test_custom_lib.custom_op(x)

# Check forward pass correctness
expected_y = x * 2.0
torch.testing.assert_close(y, expected_y)

# Check backward pass correctness
y.sum().backward()
# The gradient should be 2.0 * gradient of the sum (which is 1.0)
expected_grad = torch.ones_like(x) * 2.0
torch.testing.assert_close(x.grad, expected_grad)