import torch
import torch.nn.functional as F

input = torch.randn(1, 1, 2, 2)
indices = torch.tensor([[[[10000, 10], [0, 2]]]], dtype=torch.int64)
output_size = (2, 2)

try:
    out_cpu = F.max_unpool2d(input, indices, output_size)
    print("CPU:", out_cpu)
except Exception as e:
    print("CPU:", e)  # RuntimeError: Found an invalid max index: 10000 (output volumes are of size 4x4

# MPS
input = input.to("mps")
indices = indices.to("mps")
try:
    out_mps = F.max_unpool2d(input, indices, output_size)
    print("MPS:", out_mps)  # return a tensor here
except Exception as e:
    print("MPS:", e)


input = torch.randn(1, 1, 2, 2, 2)

indices = torch.tensor(
    [[[
        [[10000, 10], [0, 2]],
        [[1, 3], [4, 5]]
    ]]], dtype=torch.int64
)
output_size = (4, 4, 4)

# CPU
try:
    out_cpu = F.max_unpool3d(input, indices, output_size)
    print("CPU:", out_cpu)  # RuntimeError: Found an invalid max index: 10000 (output volumes are of size 4x4x4
except Exception as e:
    print("CPU:", e)

# MPS
if torch.backends.mps.is_available():
    input_mps = input.to("mps")
    indices_mps = indices.to("mps")
    try:
        out_mps = F.max_unpool3d(input_mps, indices_mps, output_size)
        print("MPS:", out_mps)  # return a tensor here
    except Exception as e:
        print("MPS:", e)