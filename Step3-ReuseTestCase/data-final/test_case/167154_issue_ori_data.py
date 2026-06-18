import torch
from torch.nn import functional as F

weight = torch.rand((768, 768), device="mps", dtype=torch.float32)
bias = torch.rand((768), device="mps", dtype=torch.float32)

shape = (5, 499, 768)
stride = (0, 768, 1)
storage_offset = 0
numel = storage_offset + sum((shape[i] - 1) * stride[i] for i in range(len(shape))) + 1
base = torch.arange(numel, dtype=torch.float32, device="mps")
input = torch.as_strided(base, size=shape, stride=stride, storage_offset=storage_offset)

F.linear(input, weight, bias)