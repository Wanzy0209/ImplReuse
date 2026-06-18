import torch

# Replicate the "fails" scenario from the bug report
# Large tensors, complex64, mps
a = torch.rand((64, 300), dtype=torch.complex64, device="mps")
b = torch.rand((64, 10000), dtype=torch.complex64, device="mps")
c = torch.rand((10000, 300), dtype=torch.complex64, device="mps")

# Adaptation: Test torch.amin on the large tensor 'b'
# Original API: torch.addmm(a, b, c, alpha=1.0, beta=0.5)
# Similar API: torch.amin(input)

mps_result = torch.amin(b)
cpu_result = torch.amin(b.cpu())

torch.testing.assert_close(mps_result.cpu(), cpu_result)