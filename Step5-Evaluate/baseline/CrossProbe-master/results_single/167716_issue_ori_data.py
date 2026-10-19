# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
a = torch.tensor([[1., 0, 2], [0, 3, 0]]).to_sparse().requires_grad_()
b = torch.tensor([[0, 1.], [2, 0], [0, 0]], requires_grad=True)
y = torch.sparse.mm(a, b)
z = y.to_dense()