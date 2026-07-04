import torch

def f_like(input_tensor: torch.Tensor) -> torch.Tensor:
    # Using torch.randn_like as the target API
    return torch.randn_like(input_tensor)

B, F, T = 1, 641, 39

src = torch.randn(B, F, T)
mismatch = src.permute(0, 2, 1)

# Handle environments where torch.compile is not available (PyTorch < 2.0)
if hasattr(torch, 'compile'):
    compiled = torch.compile(f_like, fullgraph=True)
else:
    # Fallback for older PyTorch versions: use the function directly
    compiled = f_like

_ = compiled(src)
_ = compiled(mismatch)