# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch

@torch.compile(fullgraph=True)
def f(x):
    torch._check(x.shape[0] > 3, lambda: f"{x.shape[0]} is not greater than 3")
    return x + 1

x = torch.randn(3, device="cuda")
torch._dynamo.maybe_mark_dynamic(x, 0)
print(f(x))