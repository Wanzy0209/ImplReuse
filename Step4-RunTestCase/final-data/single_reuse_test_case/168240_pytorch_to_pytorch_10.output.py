import torch

# Create a library object to define and implement the custom operator
# This is the standard way to define custom ops in PyTorch
lib = torch.library.Library("custom_namespace", "DEF")

# Define the operator schema
lib.define("my_op(Tensor x) -> Tensor")

# Define the implementation function
def my_op_impl(x):
    return x * 2.0

# Implement the operator for CPU
lib.impl("my_op", my_op_impl, "CPU")

# Verify the operator works correctly
x = torch.rand((1, 3, 224, 224))
result = torch.ops.custom_namespace.my_op(x)
expected = x * 2.0
torch.testing.assert_close(result, expected)