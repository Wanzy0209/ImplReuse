import torch

# Mock torch.compile if not available (for PyTorch < 2.0)
if not hasattr(torch, 'compile'):
    torch.compile = lambda func: func

# Setup inputs: Matrix (1024, 1024) and Vector (1024)
# Using float16 to match the original bug scenario
A = torch.rand((1024, 1024), device="cuda", dtype=torch.float16)
B = torch.rand((1024), device="cuda", dtype=torch.float16)

@torch.compile
def mv_func(mat, vec):
    # Call torch.mv with out_dtype argument
    return torch.mv(mat, vec, out_dtype=torch.float32)

# Execute
result = mv_func(A, B)

# Verify output dtype
assert result.dtype == torch.float32