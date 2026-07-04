import torch

# Handle environments where torch.compile is not available (PyTorch < 2.0)
if not hasattr(torch, 'compile'):
    # Mock torch.compile to return the function as-is
    # This preserves the test logic of running the function with different inputs
    # even if the compilation step is not supported.
    torch.compile = lambda func, **kwargs: func

def f(input: torch.Tensor) -> torch.Tensor:
    # Adapted to use torch.renorm instead of torch.complex
    # p=2 (Euclidean norm), dim=1, maxnorm=1.0
    return torch.renorm(input, p=2, dim=1, maxnorm=1.0)

B, F, T = 1, 641, 39

# Create source tensor with shape (1, 641, 39)
src = torch.randn(B, F, T)

# Create mismatched tensor with shape (1, 39, 641)
mismatch = src.permute(0, 2, 1)

# Compile the function with fullgraph=True to match the original bug scenario
compiled = torch.compile(f, fullgraph=True)

# First run with the original shape
_ = compiled(src)

# Second run with the permuted shape to test for the crash
_ = compiled(mismatch)