import torch
from torch.library import Library, register_fake

# Setup: Reproduce the tensor 'a' from the original bug report
# to ensure we have a tensor with specific stride properties.
A = torch.rand(5, 5)
Q, R = torch.linalg.qr(A)
rhs = torch.ones(Q.shape[0], 1)
a = torch.linalg.solve_triangular(R, Q.T @ rhs, upper=True)

# Define a custom library and operator to test the abstract implementation mechanism
lib = Library("test_stride_lib", "DEF")
lib.define("custom_clone_preserve(Tensor x) -> Tensor")

# Use torch.library.register_fake (the implementation of torch.library.impl_abstract)
# to define the abstract behavior. This is the API under test.
# We explicitly ensure the abstract implementation preserves strides.
@register_fake("test_stride_lib::custom_clone_preserve")
def custom_clone_preserve_abstract(x):
    # The bug in torch.compile was that the abstract impl of clone didn't preserve strides.
    # Here we verify that we can define an abstract impl that correctly handles
    # memory_format=torch.preserve_format.
    return x.clone(memory_format=torch.preserve_format)

# Register the concrete implementation for eager execution
lib.impl("test_stride_lib::custom_clone_preserve", lambda x: x.clone(memory_format=torch.preserve_format))

# Test function using the custom operator
def f(x):
    return torch.ops.test_stride_lib.custom_clone_preserve(x)

# 1. Eager execution
res_eager = f(a)
eager_stride = res_eager.stride()

# 2. Compiled execution (relies on the abstract implementation registered above)
f_compiled = torch.compile(f)
res_compiled = f_compiled(a)
compiled_stride = res_compiled.stride()

# Verify that the abstract implementation (used by compile) preserves strides
# just like the eager implementation.
assert eager_stride == compiled_stride, (
    f"Test Failed: Strides mismatch.\n"
    f"Eager stride: {eager_stride}\n"
    f"Compiled stride: {compiled_stride}\n"
    f"This indicates the abstract implementation did not preserve the format."
)

print("Test passed: torch.library.register_fake correctly preserves strides in compiled mode.")