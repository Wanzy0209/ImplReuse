import torch
import torch.nn.functional as F

# Reproduce the issue with simplified matrix inversion
B, N, D = 2, 10, 4
x = torch.randn(B, D, N)
mask = torch.ones(B, 1, N)

# Create XTX matrix similar to original code
x_expanded = (x*mask).unsqueeze(3)
x_expanded_T = x_expanded.transpose(2, 3)
XTX = torch.matmul(x_expanded, x_expanded_T)

# Test on CPU
XTX_cpu = XTX.cpu()
XTX_inv_cpu = torch.linalg.inv(XTX_cpu)
print('CPU inverse shape:', XTX_inv_cpu.shape)
print('CPU inverse values:', XTX_inv_cpu[0, 0])

# Test on MPS if available
if torch.backends.mps.is_available():
    XTX_mps = XTX.mps()
    XTX_inv_mps = torch.linalg.inv(XTX_mps)
    print('MPS inverse shape:', XTX_inv_mps.shape)
    print('MPS inverse values:', XTX_inv_mps[0, 0])
    print('Difference:', torch.max(torch.abs(XTX_inv_cpu - XTX_inv_mps.cpu())))