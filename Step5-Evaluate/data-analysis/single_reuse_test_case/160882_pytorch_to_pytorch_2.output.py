import torch

# Check if torch.compile exists (available in PyTorch 2.0+)
# If not, define a no-op wrapper to allow the test to run in older environments
if not hasattr(torch, 'compile'):
    torch.compile = lambda func, **kwargs: func

def f(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Adapted to use torch.equal instead of torch.complex
    return torch.equal(a, b)

B, F, T = 1, 641, 39

a_src = torch.randn(B, F, T)
b_src = torch.randn(B, F, T)

# Create inputs with different shapes (permuted) to test dynamic behavior
a_mismatch = a_src.permute(0, 2, 1)
b_mismatch = b_src.permute(0, 2, 1)

# Compile with fullgraph=True as in the original bug report
compiled = torch.compile(f, fullgraph=True)

# First call with original shapes
res1 = compiled(a_src, b_src)
assert res1 == False

# Second call with permuted shapes
# This verifies that torch.equal handles shape changes correctly under torch.compile
res2 = compiled(a_mismatch, b_mismatch)
assert res2 == False