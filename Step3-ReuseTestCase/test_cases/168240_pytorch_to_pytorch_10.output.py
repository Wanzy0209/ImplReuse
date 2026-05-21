import torch
from torch.library import define, impl

# Define a custom operator using the similar API
define("custom_namespace::my_op", "(Tensor x) -> Tensor")

# Implement the operator for CPU
@impl("custom_namespace::my_op", "CPU")
def my_op_impl(x):
    return x * 2.0

# Verify the operator works correctly
x = torch.rand((1, 3, 224, 224))
result = torch.ops.custom_namespace.my_op(x)
expected = x * 2.0
torch.testing.assert_close(result, expected)