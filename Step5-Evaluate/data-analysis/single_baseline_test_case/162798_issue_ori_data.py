# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch

x = torch.empty((0,), dtype=torch.float32)
print("Input tensor:", x)

# nanmedian on CPU
cpu_result = torch.nanmedian(x)
print("CPU result:", cpu_result) # tensor(nan)

# nanmedian on CUDA
if torch.cuda.is_available():
    cuda_x = x.to('cuda')
    cuda_result = torch.nanmedian(cuda_x)
    print("CUDA result:", cuda_result) # tensor(nan, device='cuda:0')

# nanmedian on MPS
if torch.backends.mps.is_available():
    mps_x = x.to('mps')
    mps_result = torch.nanmedian(mps_x)
    print("MPS result:", mps_result) # tensor(0, device='mps:0')