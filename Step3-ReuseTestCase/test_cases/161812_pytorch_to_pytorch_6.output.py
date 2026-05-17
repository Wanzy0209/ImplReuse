import torch as th

# Setup from the original bug report
x = th.nested.nested_tensor([th.ones(3, 2, 3), th.ones(4, 2, 3)], layout=th.jagged)

# Test the similar API: torch.isnan
# Adapted from the original failing call th.cat([x, x])
result = th.isnan(x)

# Verify the result is a tensor and has the expected boolean dtype
assert isinstance(result, th.Tensor)
assert result.dtype == th.bool