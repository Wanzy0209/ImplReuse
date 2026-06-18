import torch
import torch.nn as nn

# Create a simple module
m = nn.Linear(5, 7)

# Define a backward hook function
def backward_hook(module, grad_input, grad_output):
    print("Backward hook triggered")

# Register the backward hook
handle = m.register_module_backward_hook(backward_hook)

# Perform a forward and backward pass to trigger the hook
input_tensor = torch.randn(1, 5)
output = m(input_tensor)
output.sum().backward()

# Remove the hook using the handle (similar to prune.remove)
handle.remove()