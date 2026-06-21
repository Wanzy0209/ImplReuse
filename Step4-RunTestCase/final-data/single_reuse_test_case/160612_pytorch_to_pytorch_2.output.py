import torch

# Prepare a list of tensors with the same dimensionality but different sizes
# This is the typical use case for torch.nested.nested_tensor
tensor_list = [
    torch.randn(2, 3),
    torch.randn(4, 3),
    torch.randn(1, 3)
]

# Call the similar API
nt = torch.nested.nested_tensor(tensor_list)

# Verify the result is a nested tensor
assert isinstance(nt, torch.Tensor)
assert nt.is_nested

# Verify basic properties
# A list of 2D tensors results in a 3D nested tensor (1 dim for the list + 2 dims for the tensors)
assert nt.dim() == 3
assert nt.size(0) == 3  # Number of tensors in the list

print(nt)