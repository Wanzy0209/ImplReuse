import torch
import torch.library

# Define a dummy library and operator to allow torch.library.fallthrough_kernel to be called
def dummy_impl(x):
    return x

torch.library.define("test::dummy_op", "(Tensor) -> Tensor")
torch.library.impl("test::dummy_op", "CPU", dummy_impl)

def fn(x, n):
    if n == 0:
        return x
    return fn(x, n - 1) + 1

@torch.compile(backend="eager")
def outer(x):
    return fn(x, 1000)

# Replacement: sys.setrecursionlimit -> torch.library.fallthrough_kernel
# Original: sys.setrecursionlimit(10000000)
torch.library.fallthrough_kernel("test::dummy_op", "CPU")

# Replacement: torch.ones -> torch.dist
# Original: outer(torch.ones(3))
# We construct inputs for torch.dist without using torch.ones
t1 = torch.tensor([1.0, 1.0, 1.0])
t2 = torch.tensor([0.0, 0.0, 0.0])
input_tensor = torch.dist(t1, t2)

# Since we replaced the recursion limit increase with a kernel registration,
# the default recursion limit applies. We expect a RecursionError.
try:
    outer(input_tensor)
    # If the system has a high default limit, this might pass, but typically 1000 depth hits the limit.
    # We assert False if it succeeds to indicate the replacement didn't preserve the "fix" logic.
    assert False, "Expected RecursionError but execution succeeded"
except RecursionError:
    # This is the expected outcome of the replacement, as the limit was not actually raised.
    pass