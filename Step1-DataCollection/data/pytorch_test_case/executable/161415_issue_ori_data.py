import torch

input_tensor  = torch.empty_strided(size = [1792, 1899, 160], stride = (303872, 160, 1))
output_tensor = torch.rand_like(input_tensor)
assert input_tensor.stride() == output_tensor.stride()