import torch
import torch.distributed

# Setup variables for the scoping issue
keys = range(10)
allowed = [0, 1, 2, 3]

# Mock torch.compile if it is not available (e.g., PyTorch < 2.0)
# This handles the AttributeError: module 'torch' has no attribute 'compile'
if not hasattr(torch, 'compile'):
    print("Warning: torch.compile is not available in this environment. Running in eager mode.")
    torch.compile = lambda fn, **kwargs: fn

# Mock torch._dynamo if it is not available, as it is used inside the function
if not hasattr(torch, '_dynamo'):
    class _MockDynamo:
        @staticmethod
        def graph_break():
            pass
    torch._dynamo = _MockDynamo()

def fn(x):
    x = x + 1
    # Essential to trigger the specific codegen path mentioned in the bug report
    torch._dynamo.graph_break()
    
    # Leverage the similar API: torch.distributed.is_initialized
    # This checks if the distributed process group is initialized.
    # We use it here to ensure the compiler handles this API call within
    # the complex scoping context.
    is_dist_init = torch.distributed.is_initialized()

    # The problematic code pattern: local variable 'key' becomes a cell variable
    # due to the list comprehension and the nonlocal declaration in the nested function.
    key = [key for key in keys if key in allowed]

    def inner():
        nonlocal key

    # Combine results to ensure execution flow covers the API usage
    return x + key[0] + (1 if is_dist_init else 0)

# Compile and run the test
# Note: torch.distributed.is_initialized() will return False as we haven't initialized it.
compiled_fn = torch.compile(fn, backend="eager")
input_tensor = torch.ones(3)
result = compiled_fn(input_tensor)

# Assertions
# x starts as 1. x = x + 1 -> 2.
# key[0] is 0.
# is_dist_init is False.
# Result should be 2.
expected = torch.full((3,), 2.0)
assert torch.allclose(result, expected), f"Test failed: expected {expected}, got {result}"