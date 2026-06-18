import torch
import numpy as np

torch.set_default_device("cuda")

batch, in_dim, out_dim = 128, 1024, 4096
# Create input tensor
w = torch.randn(out_dim, in_dim, dtype=torch.float)
# Create index tensor for take operation
indices = torch.randint(0, out_dim * in_dim, (batch, out_dim), dtype=torch.long)

# Test torch.take
out_torch = torch.take(w, indices)
print("torch.take shape:", out_torch.shape, "stride:", out_torch.stride())

# Test numpy.take for comparison
out_np = np.take(w.cpu().numpy(), indices.cpu().numpy())
print("numpy.take shape:", out_np.shape, "stride:", out_np.strides)

# Verify contiguity (torch.take should produce contiguous output like numpy.take)
assert out_torch.is_contiguous(), "torch.take output is not contiguous"