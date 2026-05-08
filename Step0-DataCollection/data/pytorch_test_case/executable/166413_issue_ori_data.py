# pyright: strict
import torch

@torch.no_grad()
def double(x: torch.Tensor) -> torch.Tensor:
    return 2 * x

reveal_type(double)