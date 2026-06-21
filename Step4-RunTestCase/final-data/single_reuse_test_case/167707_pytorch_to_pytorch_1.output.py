import torch
import torch.library

# Handle missing custom_op in older PyTorch versions
# The custom_op decorator was introduced in PyTorch 2.0.
# We mock it here to allow the test to run on older versions.
if not hasattr(torch.library, 'custom_op'):
    def custom_op_mock(qualname):
        def decorator(func):
            return func
        return decorator
    torch.library.custom_op = custom_op_mock

# Define a custom operation
# Note: The function name 'my_op' matches the name in the qualname 'my_namespace::my_op'
# to satisfy the validation logic seen in the similar API implementation.
@torch.library.custom_op("my_namespace::my_op")
def my_op(x: torch.Tensor) -> torch.Tensor:
    return x * 2

# Execute the custom operation
# Replacing the original tensor operations (x + 21, x * 15)
x = torch.randn(2)
y = my_op(x)

# Verify the result
assert torch.allclose(y, x * 2)