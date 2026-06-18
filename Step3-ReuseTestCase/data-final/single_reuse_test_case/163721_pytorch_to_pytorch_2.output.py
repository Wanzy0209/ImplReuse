import torch
import torch.nn.functional as F

# Check for MPS availability as per the original bug context
assert torch.backends.mps.is_available()
mps_device = torch.device("mps")

# Create input tensor on MPS device
# Using dimensions similar to the original bug report (batch_size=10, features=784)
input_tensor = torch.randn(10, 784, device=mps_device)

# Call the similar API: torch.nn.functional.softmin
# softmin requires a dim argument. We use dim=1 to apply it across the feature dimension.
output = F.softmin(input_tensor, dim=1)

# Verify the result to ensure no segfault and correct execution
assert output is not None
assert output.device.type == "mps"
assert output.shape == input_tensor.shape
# Verify mathematical property of softmin: sum along dim should be 1
assert torch.allclose(output.sum(dim=1), torch.ones(10, device=mps_device))

print("Test passed.")