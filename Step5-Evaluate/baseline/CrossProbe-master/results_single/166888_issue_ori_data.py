# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
def f(x, max_val):
    y = torch.clamp(x, 0, max_val.item())
    return y

compiled_func = torch.compile(f, backend='inductor', fullgraph=True)
x = torch.randn(10, 20, 30, device='cuda')
max_val = torch.tensor(5.0, device='cuda')
compiled_func(x, max_val)