import torch

# Check for torch.compile availability (introduced in PyTorch 2.0)
if not hasattr(torch, 'compile'):
    # Mock torch.compile for compatibility with older PyTorch versions
    # It acts as a pass-through wrapper to preserve test logic
    def _mock_compile(func, **kwargs):
        return func
    torch.compile = _mock_compile

def f_like(input_tensor: torch.Tensor) -> torch.Tensor:
    # Adapted to use torch.randint_like, which takes the input tensor's shape as a reference
    return torch.randint_like(input_tensor, 100)

B, F, T = 1, 641, 39

src = torch.randn(B, F, T)
mismatch = src.permute(0, 2, 1)

compiled = torch.compile(f_like, fullgraph=True)

# First call with shape (B, F, T)
_ = compiled(src)

# Second call with shape (B, T, F) to test for shape mismatch handling
_ = compiled(mismatch)