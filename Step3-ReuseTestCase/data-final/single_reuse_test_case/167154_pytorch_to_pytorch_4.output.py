import torch
from torch.nn import functional as F

# Reproduce the non-contiguous tensor setup from the bug report
shape = (5, 499, 768)
stride = (0, 768, 1)
storage_offset = 0
numel = storage_offset + sum((shape[i] - 1) * stride[i] for i in range(len(shape))) + 1
base = torch.arange(numel, dtype=torch.float32, device="mps")
input = torch.as_strided(base, size=shape, stride=stride, storage_offset=storage_offset)

# Create a target tensor for mse_loss
target = torch.rand(shape, device="mps", dtype=torch.float32)

# Call the similar API: torch.nn.functional.mse_loss
# If the MPS buffer allocation regression bug exists, this will raise an assertion error.
output = F.mse_loss(input, target)

# Assertion to verify the operation completed successfully
assert output.dim() == 0, "Output should be a scalar with default 'mean' reduction"
print("Test passed.")