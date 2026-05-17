import torch

def f(xs):
    # Using the similar API: torch.is_storage
    # This replaces the original xs.split(1, dim=0) call
    return torch.is_storage(xs)

def backend(gm, inps):
    gm.print_readable()
    return gm

if torch.cuda.is_available():
    with torch.device("cuda"):
        xs = torch.randn(2, 2, device="cuda")

        # Eager works
        f(xs)

        # This tests if torch.is_storage fails similarly under torch.compile
        # inside a torch.device context.
        # Original bug: module 'torch._tensor' has no attribute 'split'
        torch.compile(f, backend=backend)(xs)
else:
    print("CUDA not available, skipping test.")