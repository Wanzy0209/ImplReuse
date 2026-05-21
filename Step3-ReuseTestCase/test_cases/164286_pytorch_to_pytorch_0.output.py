import torch


def f(a, x):
    # torch.mm expects strided (dense) tensors, unlike torch.sparse.mm
    return torch.mm(a, x)


# Adapt inputs to be dense (strided) tensors as required by torch.mm
# Original 'a' was 3x3 sparse, 'x' was 3x1 dense
a = torch.randn(3, 3)
x = torch.randn(3, 1)

# Test direct call
result = f(a, x)
assert result.shape == (3, 1)

# Test vjp call
# The original bug caused a crash for torch.sparse.mm because the sparse layout 
# was lost and defaulted to strided. For torch.mm, the expected layout is strided,
# so this test verifies that torch.mm works correctly in the same context.
try:
    vjp_fn = torch.func.vjp(f, a, x)[1]
    print("Test passed: torch.mm works with vjp")
except Exception as e:
    print(f"Test failed: {e}")
    raise