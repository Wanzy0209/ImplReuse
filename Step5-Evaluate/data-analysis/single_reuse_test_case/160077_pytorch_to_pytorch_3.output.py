import torch

def f(xs, idx):
    # Replacing xs.split(1, dim=0) with xs.index_select(0, idx)
    return xs.index_select(0, idx)

def backend(gm, inps):
    gm.print_readable()
    return gm

# torch.device is not a context manager, so the 'with' statement is invalid.
# We remove the 'with' block and ensure tensors are created on the correct device explicitly.
xs = torch.randn(2, 2, device="cuda")
idx = torch.tensor([0, 1], device="cuda")

# Eager works
f(xs, idx)

# This might fail with `module 'torch._tensor' has no attribute 'index_select'`
# if the bug affects similar APIs.
torch.compile(f, backend=backend)(xs, idx)