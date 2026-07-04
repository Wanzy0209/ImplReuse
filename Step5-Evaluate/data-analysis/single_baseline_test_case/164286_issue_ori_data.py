# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch


def f(a, x):
    print(a.layout, x.layout)

    return torch.sparse.mm(a, x)


a = torch.sparse_coo_tensor(torch.tensor([[0, 1, 2], [1, 2, 0]]), [1.0, 1.0, 1.0])
x = torch.tensor([1.0, 3.0, 2.0])[:, None]
print(f(a, x))  # works fine

vjp = torch.func.vjp(f, a, x)[1]  # fails