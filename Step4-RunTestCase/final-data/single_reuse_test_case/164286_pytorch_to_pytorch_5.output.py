import torch


def f(a, x):
    # Adapted to test torch.broadcast_shapes
    # torch.broadcast_shapes takes shape tuples as input, not tensors directly.
    # We extract the shapes from the tensors a and x.
    return torch.broadcast_shapes(a.shape, x.shape)


# Setup from the original bug report
a = torch.sparse_coo_tensor(torch.tensor([[0, 1, 2], [1, 2, 0]]), [1.0, 1.0, 1.0])
x = torch.tensor([1.0, 3.0, 2.0])[:, None]

# Test the direct call
# a.shape is (3, 3) (inferred from max indices 0,1,2 in both dimensions)
# x.shape is (3, 1)
# Expected broadcasted shape is (3, 3)
result = f(a, x)
print(f"Broadcasted shape: {result}")
assert result == torch.Size([3, 3]), f"Expected torch.Size([3, 3]), got {result}"

# Note: torch.func.vjp is not applicable here because torch.broadcast_shapes
# returns a torch.Size object, not a Tensor, and gradients cannot be computed
# with respect to shape tuples.