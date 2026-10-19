import torch

# logdet requires a square matrix, so we use (0, 0) instead of (0,)
x = torch.empty((0, 0), dtype=torch.float32)
print("Input tensor:", x)

# logdet on CPU
cpu_result = torch.logdet(x)
print("CPU result:", cpu_result)

# logdet on CUDA
if torch.cuda.is_available():
    cuda_x = x.to('cuda')
    cuda_result = torch.logdet(cuda_x)
    print("CUDA result:", cuda_result)

# logdet on MPS
if torch.backends.mps.is_available():
    mps_x = x.to('mps')
    mps_result = torch.logdet(mps_x)
    print("MPS result:", mps_result)