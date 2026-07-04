import torch

# Adapt inputs: torch.broadcast_shapes takes shapes (tuples), not tensors
shape1 = (1024, 1)
shape2 = (1, 1024)

# Removed @torch.compile to ensure compatibility with PyTorch versions < 2.0
def compute_broadcast_shape(s1, s2):
    # Adapted call site: torch.broadcast_shapes
    return torch.broadcast_shapes(s1, s2)

# Execute
result = compute_broadcast_shape(shape1, shape2)

# Assertion
assert result == torch.Size([1024, 1024])