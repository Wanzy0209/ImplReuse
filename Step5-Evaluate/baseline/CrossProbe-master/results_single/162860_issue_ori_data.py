# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch


def inner(x):
    return x + 1


@torch.compile(backend="eager")
def fn(x):
    x = inner(x)
    return inner(x)


fn(torch.ones(3))