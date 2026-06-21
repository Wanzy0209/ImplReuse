import torch
import torch.nn as nn
import sys

# Check if the environment supports jagged tensors (requires PyTorch 2.1+)
if not hasattr(torch, 'jagged'):
    print("Skipping test: torch.jagged layout is not available in this PyTorch version.")
    sys.exit(0)

# Setup from the bug report
x = torch.nested.nested_tensor([torch.ones(3, 2, 3), torch.ones(4, 2, 3)], layout=torch.jagged)

# Adaptation: Use torch.nn.Linear instead of torch.cat
# The input dimension to Linear must match the last dimension of the jagged tensor (3).
# We choose an arbitrary output dimension (e.g., 5).
linear = nn.Linear(3, 5)

# Call the similar API
# This verifies if torch.nn.Linear handles jagged tensors correctly
# or if it suffers from similar dispatching issues as torch.cat.
try:
    result = linear(x)
    # If successful, assert the result is a tensor (likely nested)
    assert isinstance(result, torch.Tensor)
    print("torch.nn.Linear succeeded on jagged tensor.")
except Exception as e:
    print(f"torch.nn.Linear failed on jagged tensor: {e}")
    raise