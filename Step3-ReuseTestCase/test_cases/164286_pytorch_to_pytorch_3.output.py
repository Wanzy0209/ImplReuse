import torch


def f(a, x):
    print(a.layout, x.layout)
    return torch.mv(a, x)


# Create a sparse matrix a
a = torch.sparse_coo_tensor(torch.tensor([[0, 1, 2], [1, 2, 0]]), [1.0, 1.0, 1.0])
# Create a dense vector x (1D for torch.mv)
x = torch.tensor([1.0, 3.0, 2.0])

print("Direct call:")
print(f(a, x))

print("\nVJP call:")
vjp = torch.func.vjp(f, a, x)[1]