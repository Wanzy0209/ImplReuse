import torch
import os

# Set environment variable to enable the specific logs mentioned in the issue
os.environ["TORCH_LOGS"] = "trace_bytecode"

# Check if torch.compile is available (PyTorch 2.0+)
# If not, define a mock decorator to allow the test to run without crashing
if not hasattr(torch, "compile"):
    # Mock torch.compile as a pass-through decorator for older versions
    torch.compile = lambda backend=None: lambda func: func

# Define a function using the similar API: torch.prod
# We keep torch.compile because the bug is specific to LazyVariableTracker logs within Dynamo
@torch.compile(backend="eager")
def fn(x):
    return torch.prod(x)

# Execute the function
input_tensor = torch.ones(3)
result = fn(input_tensor)

# Verify the result is correct
expected = torch.prod(input_tensor)
assert torch.equal(result, expected), f"Expected {expected}, but got {result}"