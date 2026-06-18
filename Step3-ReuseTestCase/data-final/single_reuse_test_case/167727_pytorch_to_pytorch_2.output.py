import torch

# Adapted test case for torch.nonzero based on the large tensor issue
# Using the dimensions and dtype from the failing case of the original bug report
x = torch.rand((64, 10000), dtype=torch.complex64, device="mps")

# Call torch.nonzero
out_mps = torch.nonzero(x)

# Verify against CPU implementation
torch.testing.assert_close(out_mps, x.cpu().nonzero())