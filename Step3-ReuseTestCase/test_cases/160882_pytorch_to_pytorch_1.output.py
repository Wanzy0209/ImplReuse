import torch

def f(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    # Adapted from torch.complex to torch.block_diag
    return torch.block_diag(t1, t2)

# Dimensions adapted for 2D inputs (torch.block_diag requires <= 2D)
B, F, T = 1, 641, 39

t1_src = torch.randn(B, F)
t2_src = torch.randn(B, T)

# Permute dimensions to create shape mismatch
t1_mismatch = t1_src.permute(1, 0)
t2_mismatch = t2_src.permute(1, 0)

compiled = torch.compile(f, fullgraph=True)

# First call with original shapes
_ = compiled(t1_src, t2_src)
# Second call with permuted shapes (potential crash site)
_ = compiled(t1_mismatch, t2_mismatch)