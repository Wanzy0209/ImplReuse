# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
import time

warmup = 128
iters = 16384

a = torch.zeros(512, 512, device='cuda', dtype=torch.bfloat16)
for _ in range(warmup):
    torch.matmul(a, a)

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(iters):
    torch.matmul(a, a)
torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"{1e6 * (t1 - t0)/iters}")