import torch
A = torch.randn(3, 4)
print(A.sum(-1)[0])  # sum of first row
print(A[0,:].sum())  # correct equivalent
print(A[:,0].sum())  # incorrect equivalent shown in docs