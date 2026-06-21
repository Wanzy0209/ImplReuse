import torch
from torch import tensor

# Test case for torch.nn.LazyLinear
# Adapted from the torch.aminmax issue context to verify similar API behavior

# Instantiate the LazyLinear layer
# Note: LazyLinear infers in_features from the input tensor, so only out_features is required here.
layer = torch.nn.LazyLinear(out_features=2)

# Create an input tensor
# The original issue used a 1D tensor [1, -3, 5]. 
# We adapt this to a 2D tensor suitable for a Linear layer input.
input_tensor = torch.tensor([[1.0, -3.0, 5.0]])

# Perform the forward pass
output = layer(input_tensor)

# Verify the output
# The original bug was about the return type representation being invalid code.
# Here we verify the output is a valid Tensor and the operation completes successfully.
assert isinstance(output, torch.Tensor)
assert output.shape == (1, 2), f"Expected shape (1, 2), got {output.shape}"

# Check the representation of the output to ensure it is valid (unlike the original bug)
# This should not raise a TypeError.
print(repr(output))