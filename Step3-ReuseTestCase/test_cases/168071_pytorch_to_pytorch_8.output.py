import torch as pt

# Create a tensor with a 0-shape dimension, similar to the bug report
x0 = pt.zeros((6, 0))

# Adapt the call to torch.special.gammaincc
# Note: gammaincc takes two inputs. We pass x0 for both to test handling of 0-shape tensors.
x1 = pt.special.gammaincc(x0, x0)

# Verify the output shape is preserved and no RuntimeError occurs
print(x1.shape)
assert x1.shape == (6, 0)