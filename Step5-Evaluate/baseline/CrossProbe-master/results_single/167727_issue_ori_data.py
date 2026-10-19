# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch

# success
a = torch.rand((64, 300), dtype=torch.complex64, device="mps")
b = torch.rand((64, 100), dtype=torch.complex64, device="mps")
c = torch.rand((100, 300), dtype=torch.complex64, device="mps")
out = torch.addmm(a, b, c, alpha=1.0, beta=0.5)
torch.testing.assert_close(torch.addmm(a, b, c, alpha=1.0, beta=0.5).cpu(), 
                           torch.addmm(a.cpu(), b.cpu(), c.cpu(), alpha=1.0, beta=0.5))

# fails
a = torch.rand((64, 300), dtype=torch.complex64, device="mps")
b = torch.rand((64, 10000), dtype=torch.complex64, device="mps")
c = torch.rand((10000, 300), dtype=torch.complex64, device="mps")
out = torch.addmm(a, b, c, alpha=1.0, beta=0.5)
torch.testing.assert_close(torch.addmm(a, b, c, alpha=1.0, beta=0.5).cpu(), 
                           torch.addmm(a.cpu(), b.cpu(), c.cpu(), alpha=1.0, beta=0.5))