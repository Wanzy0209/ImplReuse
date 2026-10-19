# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch


@torch.compile(backend="eager")
def fn(x, i):
    if i == 1:
        torch._dynamo.graph_break()
    return x + 1


inp = torch.randn(3)
fn(inp, 0)
fn(inp, 1)
fn(inp, 2)