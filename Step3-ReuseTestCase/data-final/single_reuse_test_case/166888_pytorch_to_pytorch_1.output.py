import torch
import functools

# Adapted test case based on the bug report and the similar API (functools.reduce)
# The original bug involved a NameError when using .item() on a float tensor arg inside torch.compile.
# We test if functools.reduce (which uses a polyfill in torch._dynamo) handles this correctly.

def f(x, max_val):
    # Use functools.reduce with .item() on the tensor argument
    # This mimics the usage pattern in the original bug report (max_val.item())
    return functools.reduce(lambda acc, val: acc + val, x, max_val.item())

# Compile the function using the inductor backend and fullgraph
compiled_func = torch.compile(f, backend='inductor', fullgraph=True)

# Setup inputs
x = torch.randn(10, 20, 30, device='cuda')
max_val = torch.tensor(5.0, device='cuda')

# Execute the compiled function
try:
    result = compiled_func(x, max_val)
    
    # Verify the result against the eager execution
    expected = functools.reduce(lambda acc, val: acc + val, x, max_val.item())
    
    # Check if results are close
    assert torch.allclose(result, expected), "Results do not match"
    print("Test passed: functools.reduce works correctly with torch.compile and .item()")
except Exception as e:
    print(f"Test failed with error: {e}")