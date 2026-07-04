import torch
import sys

# Check if torch.compile is available (requires PyTorch >= 2.0)
if not hasattr(torch, 'compile'):
    print("Skipping test: torch.compile is not available (requires PyTorch >= 2.0).")
    sys.exit(0)

# Define a function that uses torch.any in a differentiable context
# to mimic the structure of the original test case (forward + backward)
def any_func(x):
    # Create a mask using torch.any
    # Check if any element in the last dimension is positive
    mask = torch.any(x > 0, dim=-1, keepdim=True)
    # Apply the mask to the input to allow gradient flow
    return x * mask.float()

# Compile the function with the same backend settings as the bug report
compiled_any = torch.compile(any_func, fullgraph=True, backend="inductor")

with torch.device("cuda"):
    # Create input tensor with similar characteristics to the bug report
    x = torch.randn([2, 32, 4096, 128], dtype=torch.bfloat16, requires_grad=True)

# Execute forward pass
y = compiled_any(x)

# Execute backward pass to verify compilation of the backward graph
y.backward(torch.randn_like(y))

# Basic assertion to verify execution
assert y.shape == x.shape