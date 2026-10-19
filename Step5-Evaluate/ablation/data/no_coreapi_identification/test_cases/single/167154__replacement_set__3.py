import torch
from torch.nn import functional as F

# Replacing torch.rand with torch.tanh
# We use torch.randn to generate the input for torch.tanh to maintain tensor creation
weight = torch.tanh(torch.randn((768, 768), device="mps", dtype=torch.float32))
bias = torch.tanh(torch.randn((768), device="mps", dtype=torch.float32))

shape = (5, 499, 768)
stride = (0, 768, 1)
storage_offset = 0

# Replacing torch.arange with torch.dist
# We calculate the expected number of elements and use torch.dist to derive the value
# to preserve the logic of determining the buffer size requirement.
calculated_numel = storage_offset + sum((shape[i] - 1) * stride[i] for i in range(len(shape))) + 1
numel = int(torch.dist(torch.tensor(0.0), torch.tensor(float(calculated_numel))))

# Replacing torch.as_strided with torch.empty_strided
# torch.empty_strided allocates a new tensor with the specified shape and stride,
# directly testing the MPS backend's buffer allocation for non-contiguous tensors.
input = torch.empty_strided(shape, stride, device="mps", dtype=torch.float32)

# Run the operation
output = F.linear(input, weight, bias)

# Assertion to verify the operation completed successfully and output shape is correct
assert output.shape == (5, 499, 768)