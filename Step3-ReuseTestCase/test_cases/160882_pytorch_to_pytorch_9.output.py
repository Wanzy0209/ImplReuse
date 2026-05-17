import torch

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