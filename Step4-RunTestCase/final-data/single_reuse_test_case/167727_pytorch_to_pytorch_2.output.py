import torch

# Adapted test case for torch.nonzero based on the large tensor issue
# Using the dimensions from the failing case of the original bug report
# Note: Changed dtype from complex64 to float32 as MPS backend does not support complex64 for torch.rand
x = torch.rand((64, 10000), dtype=torch.float32, device="mps")

# Call torch.nonzero
out_mps = torch.nonzero(x)

# Verify against CPU implementation
torch.testing.assert_close(out_mps, x.cpu().nonzero())