# MOCK_TRANSLATION to tensorflow
# original source preserved below
# pyright: strict
import torch

@torch.no_grad()
def double(x: torch.Tensor) -> torch.Tensor:
    return 2 * x

reveal_type(double)