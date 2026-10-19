import torch

# Create an empty square matrix (0x0) for logdet
# Note: logdet requires a square matrix, unlike nanmedian which takes any tensor
x = torch.empty((0, 0), dtype=torch.float32)
print("Input tensor:", x)

# logdet on CPU
cpu_result = torch.logdet(x)
print("CPU result:", cpu_result)
assert cpu_result.item() == 0.0, f"Expected 0.0 on CPU, got {cpu_result.item()}"

# logdet on CUDA
if torch.cuda.is_available():
    cuda_x = x.to('cuda')
    cuda_result = torch.logdet(cuda_x)
    print("CUDA result:", cuda_result)
    assert cuda_result.item() == 0.0, f"Expected 0.0 on CUDA, got {cuda_result.item()}"

# logdet on MPS
if torch.backends.mps.is_available():
    mps_x = x.to('mps')
    mps_result = torch.logdet(mps_x)
    print("MPS result:", mps_result)
    assert mps_result.item() == 0.0, f"Expected 0.0 on MPS, got {mps_result.item()}"