import torch

def f(xs):
    # Adapted to use the functional torch.split API instead of the method xs.split
    return torch.split(xs, 1, dim=0)

def backend(gm, inps):
    gm.print_readable()
    return gm

# The bug is triggered specifically inside a torch.device context
if torch.cuda.is_available():
    with torch.device("cuda"):
        xs = torch.randn(2, 2, device="cuda")

        # Eager execution
        f(xs)

        # Compiled execution
        # Bug: 'torch._tensor' has no attribute 'split'
        torch.compile(f, backend=backend)(xs)
else:
    # Fallback for environments without CUDA to test logic
    with torch.device("cpu"):
        xs = torch.randn(2, 2, device="cpu")
        f(xs)
        torch.compile(f, backend=backend)(xs)