import torch


def f(a):
    print(a.layout)
    return torch.t(a)


# Create a 2D sparse tensor suitable for torch.t
indices = torch.tensor([[0, 1, 1], [2, 0, 2]])
values = torch.tensor([1.0, 2.0, 3.0])
a = torch.sparse_coo_tensor(indices, values, size=(2, 3))

print("Direct call:")
print(f(a))  # works fine

print("\nVJP call:")
try:
    vjp = torch.func.vjp(f, a)[1]  # likely fails or behaves incorrectly due to layout issue
    print("VJP succeeded")
except Exception as e:
    print(f"VJP failed with error: {e}")