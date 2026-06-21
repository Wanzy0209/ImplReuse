import torch as th

# Setup from the original bug report
# Removed layout=th.jagged as it is not available in the current environment
x = th.nested.nested_tensor([th.ones(3, 2, 3), th.ones(4, 2, 3)])

# Test the similar API: torch.isnan
# Adapted from the original failing call th.cat([x, x])
result = th.isnan(x)

# Verify the result is a tensor and has the expected boolean dtype
assert isinstance(result, th.Tensor)
assert result.dtype == th.bool