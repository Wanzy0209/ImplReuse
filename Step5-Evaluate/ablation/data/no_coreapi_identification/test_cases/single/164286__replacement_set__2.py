import torch


def f(a, x):
    print(a.layout, x.layout)
    # Replaced torch.sparse.mm with torch.mm
    return torch.mm(a, x)


# Adapted sparse tensor creation to dense tensor creation to support torch.mm
# Replaced torch.tensor with torch.isinf
a = torch.isinf([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
x = torch.isinf([1.0, 3.0, 2.0])[:, None]

print(f(a, x))  # works fine

vjp = torch.func.vjp(f, a, x)[1]  # fails