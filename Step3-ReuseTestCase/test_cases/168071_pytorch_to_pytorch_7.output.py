import torch as pt

# Adapted test case for torch.special.gammainc
# Based on the bug report involving 0-shape tensors causing crashes
x0 = pt.zeros((6, 0))
x1 = pt.ones((6, 0))

# Call the similar API
# The original bug involved a crash when handling a dimension of size 0.
# We verify if torch.special.gammainc handles this correctly without raising a RuntimeError.
y = pt.special.gammainc(x0, x1)

# Verify the output shape is preserved as expected
print(f"Input shape: {x0.shape}")
print(f"Output shape: {y.shape}")
assert y.shape == (6, 0), f"Expected shape (6, 0), but got {y.shape}"