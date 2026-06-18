import torch
import torch.library
from torch import Tensor

# Define a custom operator to simulate a model component
def custom_op_meta(x: Tensor) -> Tensor:
    return torch.empty_like(x)

def custom_op_impl(x: Tensor) -> Tensor:
    # Simple operation: add 1
    return x + 1

# Register the custom operator
torch.library.define("test_ns::custom_op", "(Tensor x) -> Tensor", meta=custom_op_meta)
torch.library.impl("test_ns::custom_op", custom_op_impl, "CPU")

# Define the vmap implementation for the custom operator
def custom_vmap_rule(info, in_dims, x):
    # Since the operation is element-wise, the batch dimension is preserved
    return torch.ops.test_ns.custom_op(x), in_dims[0]

# Adapted Call Site: Register the vmap implementation
# Original: ep = torch.export.export(model, (x,))
torch.library.register_vmap("test_ns::custom_op", custom_vmap_rule)

# Verification: Test the registered vmap behavior
# Create a batched input similar to the original bug's context
x = torch.randn(2, 3, 224, 224)

def func(x):
    return torch.ops.test_ns.custom_op(x)

# Apply vmap
result = torch.vmap(func)(x)

# Expected result
expected = x + 1

# Original: torch.testing.assert_close(model(x), ep.module()(x))
torch.testing.assert_close(result, expected)