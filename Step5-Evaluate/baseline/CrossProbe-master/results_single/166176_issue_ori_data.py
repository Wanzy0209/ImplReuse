# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch

def fn(x):
    torch._dynamo.graph_break()
    with torch.no_grad():
        with torch.no_grad():
            torch._dynamo.graph_break()
    return x + 1

inp = torch.ones(3)
opt_m = torch.compile(fn, backend="eager")
opt_m(inp)