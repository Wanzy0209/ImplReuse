import torch

def f_like(input_tensor: torch.Tensor) -> torch.Tensor:
    # Using torch.randn_like as the target API
    return torch.randn_like(input_tensor)

B, F, T = 1, 641, 39

src = torch.randn(B, F, T)
mismatch = src.permute(0, 2, 1)

compiled = torch.compile(f_like, fullgraph=True)

_ = compiled(src)
_ = compiled(mismatch)