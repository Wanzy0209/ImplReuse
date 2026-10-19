# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch


def func_nojit(x):
    return torch.full((2, ), x, dtype=torch.float64)


func_jit = torch.compile(func_nojit)

for func in [func_nojit, func_jit]:
    x1 = torch.tensor(5.0, dtype=torch.float64)
    x2 = torch.tensor(10.0, dtype=torch.float64)
    print(func(x1))
    print(func(x2))