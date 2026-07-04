import torch

# Handle API compatibility for different PyTorch versions
# impl_abstract was introduced in later versions (PyTorch 2.1+).
# For older versions (e.g., PyTorch 1.13 compatible with Python 3.7), we use register_fake.
try:
    from torch.library import impl_abstract, define, impl
except ImportError:
    from torch.library import define, impl, register_fake
    
    def impl_abstract(name, func):
        # register_fake is the equivalent of impl_abstract in older PyTorch versions
        return register_fake(name)(func)

# Define a custom operator to test registration behavior
define("test_ns::duplicate_reg_op", "(Tensor x) -> Tensor")

# Define the abstract (fake) implementation
def abstract_fn(x):
    return torch.empty_like(x)

# Define a concrete implementation to allow execution
def concrete_fn(x):
    return x + 1

# Simulate the "Merge Mistake" scenario where the registration logic is duplicated
# This mimics the structure in the bug report where joint_custom_pre_pass was invoked twice.

# First block (Original code)
if True:  # Simulating config.joint_custom_pre_pass is not None
    impl_abstract("test_ns::duplicate_reg_op", abstract_fn)

# ... (hypothetical other code) ...

# Second block (The duplicate code introduced by the merge mistake)
if True:  # Simulating config.joint_custom_pre_pass is not None
    impl_abstract("test_ns::duplicate_reg_op", abstract_fn)

# Register the concrete implementation
impl("test_ns::duplicate_reg_op", concrete_fn)

# Verification: Ensure the operator works correctly despite the duplicate registration
x = torch.randn(2, 2)
result = torch.ops.test_ns.duplicate_reg_op(x)
expected = x + 1

assert torch.allclose(result, expected), "Duplicate registration of abstract impl caused failure"
print("Test passed: Duplicate registration handled correctly.")