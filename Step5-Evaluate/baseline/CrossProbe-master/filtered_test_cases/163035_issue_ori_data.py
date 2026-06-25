import torch
import torch.nn.functional as F

input = torch.randn(1, 1, 2, 2)
indices = torch.tensor([[[[10000, 10], [0, 2]]]], dtype=torch.int64)
output_size = (2, 2)

# CPU raises error
try:
    F.max_unpool2d(input, indices, output_size)
except Exception as e:
    print(f"CPU error: {e}")

# MPS doesn't raise error
if torch.backends.mps.is_available():
    try:
        F.max_unpool2d(input.to("mps"), indices.to("mps"), output_size)
        print("MPS: No error raised")
    except Exception as e:
        print(f"MPS error: {e}")