import torch
# Reproduce test case for XPU float16 with requires_grad=True
if torch.xpu.is_available():
    x = torch.randn(2, 2, dtype=torch.float16, device='xpu', requires_grad=True)
    y = x * 2
    y.sum().backward()