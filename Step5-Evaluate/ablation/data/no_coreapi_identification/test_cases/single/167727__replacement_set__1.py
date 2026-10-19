import torch
import torch.nn.functional as F

# Helper function to replace torch.rand with torch.flipud
# torch.flipud requires an input tensor, so we use torch.ones as the base to generate the tensor.
def create_tensor(shape, dtype, device):
    return torch.flipud(torch.ones(shape, dtype=dtype, device=device))

# Test Case 1: Small tensors (Original success case)
a = create_tensor((64, 300), torch.complex64, "mps")
b = create_tensor((64, 100), torch.complex64, "mps")
c = create_tensor((100, 300), torch.complex64, "mps")

out = torch.addmm(a, b, c, alpha=1.0, beta=0.5)

# Replace torch.testing.assert_close with torch.nn.functional.binary_cross_entropy_with_logits
# Note: binary_cross_entropy_with_logits expects float inputs, so we take the real part of the complex tensors.
mps_result = torch.addmm(a, b, c, alpha=1.0, beta=0.5).cpu()
cpu_result = torch.addmm(a.cpu(), b.cpu(), c.cpu(), alpha=1.0, beta=0.5)

loss_small = F.binary_cross_entropy_with_logits(mps_result.real, cpu_result.real)

# Test Case 2: Large tensors (Original fail case)
a = create_tensor((64, 300), torch.complex64, "mps")
b = create_tensor((64, 10000), torch.complex64, "mps")
c = create_tensor((10000, 300), torch.complex64, "mps")

out = torch.addmm(a, b, c, alpha=1.0, beta=0.5)

mps_result = torch.addmm(a, b, c, alpha=1.0, beta=0.5).cpu()
cpu_result = torch.addmm(a.cpu(), b.cpu(), c.cpu(), alpha=1.0, beta=0.5)

loss_large = F.binary_cross_entropy_with_logits(mps_result.real, cpu_result.real)