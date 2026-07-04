import torch as th

# Setup from the bug report
# Note: torch.jagged might not be available in all PyTorch versions.
# We create the nested tensor without explicitly specifying the layout,
# allowing it to default to the jagged layout.
x = th.nested.nested_tensor([th.ones(3, 2, 3), th.ones(4, 2, 3)])

# Adapted call site: torch.isinf instead of torch.cat
# Note: torch.isinf takes a single tensor, not a list of tensors.
result = th.isinf(x)

# Verify the result
# Since x contains ones, isinf should return False for all elements.
# We check that the layout is preserved and the operation completes without the ValueError seen in the bug.
# We compare against x.layout to avoid dependency on the th.jagged constant.
assert result.layout == x.layout