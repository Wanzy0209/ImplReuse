import torch
if torch.xpu.is_available():
    # Test grid sampler 3D with float64 on XPU
    input = torch.randn(1, 1, 8, 8, 8, dtype=torch.float64).xpu()
    grid = torch.randn(1, 8, 8, 8, 3, dtype=torch.float64).xpu()
    output = torch.grid_sampler(input, grid, 0, 0, False)