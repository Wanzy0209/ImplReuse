import torch

def fn(x, n):
    if n == 0:
        return x
    return fn(x, n - 1) + 1

@torch.compile(backend="eager")
def outer(x):
    return fn(x, 1000)

# Replacement: sys.setrecursionlimit -> torch.backends.cuda.math_sdp_enabled
torch.backends.cuda.math_sdp_enabled(True)

# Replacement: torch.ones -> torch.det
# Note: torch.det requires a square matrix, so we use (3, 3)
input_tensor = torch.ones(3, 3)
det_val = torch.det(input_tensor)

# Execute
result = outer(det_val)

# Assertion to verify logic preservation (recursion adds 1 1000 times to 0.0)
assert result == 1000.0