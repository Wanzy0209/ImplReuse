import torch
# Reproduce test conditions for XPU with bfloat16 and requires_grad=True
if torch.xpu.is_available():
    x = torch.randn(2, 2, dtype=torch.bfloat16, device='xpu', requires_grad=True)
    y = torch.randn(2, 2, dtype=torch.bfloat16, device='xpu')
    z = x + y
    z.sum().backward()
else:
    print('XPU not available')