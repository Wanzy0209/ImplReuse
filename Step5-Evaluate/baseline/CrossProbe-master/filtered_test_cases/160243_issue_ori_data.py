import torch
if torch.xpu.is_available():
    # Test copy non-blocking with pinned memory on XPU
    x = torch.randn(10, device='xpu')
    y = torch.empty_like(x, pin_memory=True)
    y.copy_(x, non_blocking=True)