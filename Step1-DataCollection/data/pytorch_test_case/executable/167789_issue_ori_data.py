import sys
import torch

def fn(x, n):
    if n == 0:
        return x
    return fn(x, n - 1) + 1

@torch.compile(backend="eager")
def outer(x):
    return fn(x, 1000)

sys.setrecursionlimit(10000000)
outer(torch.ones(3))