# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
from torch import nn


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)

    return torch.log(x1 / x2)


class Model(nn.Module):
    @torch.compiler.disable(recursive=False)  # type: ignore[misc]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = inverse_sigmoid(x)
        return x


n = Model()
torch.jit.script(n)