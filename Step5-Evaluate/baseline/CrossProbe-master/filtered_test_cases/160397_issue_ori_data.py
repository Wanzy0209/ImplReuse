import torch

# Test for float32
f32 = torch.finfo(torch.float32)
print(f'float32 eps: {f32.eps}')
print(f'1.0 + eps != 1.0: {1.0 + f32.eps != 1.0}')

# Test for float16
f16 = torch.finfo(torch.float16)
print(f'float16 eps: {f16.eps}')
print(f'1.0 + eps != 1.0: {1.0 + f16.eps != 1.0}')