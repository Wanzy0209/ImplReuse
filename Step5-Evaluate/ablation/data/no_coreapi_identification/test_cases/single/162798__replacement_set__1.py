import torch

# torch.empty -> torch.isinf
# torch.float32 -> torch.logdet (Ignored, kept float32 for validity)
# Note: torch.logdet requires a square matrix, so we use (0, 0) instead of (0,)
x = torch.isinf(torch.empty((0, 0), dtype=torch.float32)).to(torch.float32)
print("Input tensor:", x)

# torch.nanmedian -> torch.logdet
cpu_result = torch.logdet(x)
print("CPU result:", cpu_result) 
# Expected: tensor(nan)

# torch.cuda.is_available -> torch.logdet (Ignored, kept check for validity)
if torch.cuda.is_available():
    # x.to -> torch.logdet (Ignored, kept to for validity)
    cuda_x = x.to('cuda')
    cuda_result = torch.logdet(cuda_x)
    print("CUDA result:", cuda_result) 
    # Expected: tensor(nan, device='cuda:0')

# torch.backends.mps.is_available -> torch.backends.cusparselt.is_available
if torch.backends.cusparselt.is_available():
    # cuSPARSELt is a CUDA library, so we run on CUDA if available, else CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cusparselt_x = x.to(device)
    cusparselt_result = torch.logdet(cusparselt_x)
    print("cuSPARSELt result:", cusparselt_result)
    # Expected: tensor(nan, device='...')

# Assertions to check for the bug (returning 0 instead of nan)
assert torch.isnan(cpu_result), "CPU result should be NaN for empty input"
if torch.cuda.is_available():
    assert torch.isnan(cuda_result), "CUDA result should be NaN for empty input"
if torch.backends.cusparselt.is_available():
    assert torch.isnan(cusparselt_result), "cuSPARSELt result should be NaN for empty input"