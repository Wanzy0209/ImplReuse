import torch

keys = range(10)
allowed = [0, 1, 2, 3]


def fn(x):
    x = x + 1
    torch._dynamo.graph_break()
    key = [key for key in keys if key in allowed]

    def inner():
        nonlocal key

    return x + key[0]


torch.compile(fn, backend="eager")(torch.ones(3))