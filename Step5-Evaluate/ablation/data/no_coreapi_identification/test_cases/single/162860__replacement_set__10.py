import torch

def inner(x):
    return x + 1

@torch.compile(backend="eager")
def fn(x):
    x = inner(x)
    return inner(x)

# Replacement: torch.ones -> torch.qr
# torch.qr computes the QR decomposition of a matrix. 
# We generate a random matrix and use the Q component as the input tensor.
input_matrix = torch.randn(3, 3)
q, _ = torch.qr(input_matrix)

fn(q)