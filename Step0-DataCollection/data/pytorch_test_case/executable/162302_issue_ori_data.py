import torch

A = torch.tensor([
    [4.0, -3.0,  2.0, -1.0],
    [-1.0,  2.0, -3.0,  4.0],
    [3.0, -4.0,  1.0, -2.0],
    [-2.0,  1.0, -4.0,  3.0]
], dtype=torch.float32)

cpu_result = torch.linalg.tensorinv(A, ind=1)
gpu_result = torch.linalg.tensorinv(A.to("cuda"), ind=1)

print("CPU:\n", cpu_result)
print("GPU:\n", gpu_result)
print("CPU / GPU elementwise ratio:\n", cpu_result / gpu_result.cpu())