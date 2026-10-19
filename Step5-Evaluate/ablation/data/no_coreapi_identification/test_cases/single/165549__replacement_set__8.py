import torch
import torch.nn.functional as F

# Setup: Create an input tensor for max_pool3d on the custom device.
# We use torch.empty to initialize the input, as torch.randn is being replaced.
input_tensor = torch.empty(1, 1, 4, 4, 4, device='privateuse1').uniform_()

# Replacement: torch.randn -> torch.nn.functional.max_pool3d
t = F.max_pool3d(input_tensor, kernel_size=2)

# Test the failing case: torch.abs
# Bug: returns tensor with shape [0]
result = torch.abs(t)
assert result.shape == t.shape, f"Bug reproduced: expected {t.shape}, got {result.shape}"

# Test the working out variant
pre_allocated_tensor = torch.empty_like(t)
torch.abs(t, out=pre_allocated_tensor)
assert pre_allocated_tensor.shape == t.shape

# Test the working in-place variant
t_inplace = t.clone()
t_inplace.abs_()
assert t_inplace.shape == t.shape

# Replacement: torch.neg -> torch.asinh
# Test the control case (expected to work)
result_asinh = torch.asinh(t)
assert result_asinh.shape == t.shape