import torch


@torch.compile(backend="eager")
def fn(x):
    y = x + 1
    z = x + y
    torch._dynamo.graph_break()
    return z


fn(torch.ones(3))