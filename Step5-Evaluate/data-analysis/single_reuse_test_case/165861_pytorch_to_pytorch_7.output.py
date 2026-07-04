import torch
import torch.special

# Reproduce the large dimension scenario from the bug report
# The bug occurs when a batch dimension is larger than uint16 max (2**16)
x = torch.rand(2**16, 2, device="cuda")
y = torch.rand(2**16, 2, device="cuda")

# Test the similar API: torch.special.gammainc
# We verify that the API handles large dimensions correctly on CUDA without crashing
out = torch.special.gammainc(x, y)
print("gammainc ok")

# Verify output shape and basic validity
assert out.shape == x.shape
assert not torch.isnan(out).all()