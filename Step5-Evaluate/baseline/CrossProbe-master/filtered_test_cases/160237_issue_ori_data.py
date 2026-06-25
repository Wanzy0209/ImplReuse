import torch
if torch.backends.mps.is_available():
    input = torch.randn(1, 1, 8, 8, 8)
    grid = torch.randn(1, 8, 8, 8, 3)
    out = torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=True)