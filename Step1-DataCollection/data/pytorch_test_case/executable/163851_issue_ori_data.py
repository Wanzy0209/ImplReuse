import torch

input = torch.ones(1,1,3,3,3)
grid_nan = torch.tensor([[[[[torch.nan, 1., 1.],[1., 1., 1.]]]]])

out_cpu = torch.grid_sampler_3d(input, grid_nan, 0, 0, True)
out_mps = torch.grid_sampler_3d(input.to("mps"), grid_nan.to("mps"), 0, 0, True)
print("CPU:", out_cpu.flatten()) # tensor([nan, 1.])
print("MPS:", out_mps.flatten()) # tensor([1., 1.], device='mps:0')