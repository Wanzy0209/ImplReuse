import torch

def f(x: torch.Tensor, factor: int) -> torch.Tensor:
    return torch.nn.functional.pixel_unshuffle(x, factor)

B, C, H, W = 1, 4, 8, 4
factor = 2

x_src = torch.randn(B, C, H, W)
# Permute H and W dimensions to change the input shape for the second call
x_mismatch = x_src.permute(0, 1, 3, 2)

compiled = torch.compile(f, fullgraph=True)

# First call with shape (1, 4, 8, 4)
_ = compiled(x_src, factor)
# Second call with shape (1, 4, 4, 8)
_ = compiled(x_mismatch, factor)