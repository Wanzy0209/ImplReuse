# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch

def f(x):
    return x.sin().cos()


@torch.compile
def g(x):
    return torch.vmap(f)(x)

x = torch.randn(10, device="cuda")
g(x)