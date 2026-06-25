import torch
if torch.xpu.is_available():
    # Reproduce grid sampler 3D test on XPU
    input = torch.randn(1, 1, 8, 8, 8, dtype=torch.float32, device='xpu')
    grid = torch.randn(1, 8, 8, 8, 3, dtype=torch.float32, device='xpu')
    output = torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=True)