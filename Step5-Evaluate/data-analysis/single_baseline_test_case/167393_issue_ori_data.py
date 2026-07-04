# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch

torch._dynamo.config.nested_graph_breaks = True


def inner(x):
    x = x + 1
    torch._dynamo.graph_break()
    return x + 2


@torch.compile(backend="eager")
def outer(x):
    x = inner(x + 4) + 8
    return inner(x) + 16


outer(torch.ones(3))