import torch
import torch.nn.functional as F


def f(a):
    print(a.layout)
    return F.softshrink(a, 0.5)


# Changed from sparse_coo_tensor to a dense tensor because
# F.softshrink is not implemented for the SparseCPU backend.
a = torch.tensor([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
print(f(a))  # works fine

vjp = torch.func.vjp(f, a)[1]  # works fine