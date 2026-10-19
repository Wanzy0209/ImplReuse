# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch

dividend = torch.full((2, 3), torch.iinfo(torch.int64).min, dtype=torch.int64, device='cpu')
divisor = torch.full((3,), -1, dtype=torch.int64, device='cpu')

print("Dividend tensor:", dividend)
print("Divisor tensor:", divisor)

result = torch.fmod(dividend, divisor)
print("Result:", result)