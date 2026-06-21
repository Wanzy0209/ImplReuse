import torch
import functools
from torch import library

# 1. Define a custom operator to test with
# Note: library.register_autograd is not a public API in PyTorch.
# The correct way to define a custom op with autograd support is using the custom_op decorator.
@library.custom_op("test_ns::custom_op", mutates_args=())
def custom_op(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x) * x

# 2. Implement the forward pass
@library.impl("test_ns::custom_op", "CPU")
def custom_op_impl(x):
    return torch.sin(x) * x

# 3. Define the backward and setup_context functions
def setup_context_fn(ctx, inputs, output):
    x, = inputs
    ctx.save_for_backward(x)

def backward_fn(ctx, grad_output):
    x, = ctx.saved_tensors
    # Derivative of sin(x)*x is x*cos(x) + sin(x)
    return grad_output * (x * torch.cos(x) + torch.sin(x))

# 4. Apply functools.partial to the backward function
# This mimics the original bug where a functools.partial'ed callback was passed.
# We are testing if register_autograd handles partial'ed callables correctly,
# especially under torch.compile.
partial_backward = functools.partial(backward_fn)

# 5. Register the autograd mechanism using the partial'ed function
# The object returned by the custom_op decorator has a register_backward method.
custom_op.register_backward(partial_backward, setup_context=setup_context_fn)

# 6. Test within a compiled context (similar to the original bug report)
@torch.compile(backend="aot_eager_decomp_partition", fullgraph=True)
def run_compiled_op(x):
    return torch.ops.test_ns.custom_op(x)

# 7. Execution and Verification
a = torch.randn(4, 4, requires_grad=True, device="cpu")

# Forward pass
output = run_compiled_op(a)

# Backward pass
try:
    output.sum().backward()
    print("Test passed: functools.partial'ed backward function handled correctly.")
except Exception as e:
    print(f"Test failed: {e}")