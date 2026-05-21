import torch
import torch.nn.utils as utils

# Setup from the original bug report, adapted for the similar API
# We add requires_grad=True because clip_grad_norm_ operates on gradients
a = torch.randn(3, requires_grad=True)
b = torch.randn(5, requires_grad=True)
nt = torch.nested.nested_tensor([a, b], layout=torch.jagged)

# To make the test for clip_grad_norm_ meaningful, we need gradients.
# We perform a dummy backward pass.
# Note: Depending on PyTorch version, specific operations on NestedTensor might vary,
# but sum() is generally supported for generating gradients.
loss = nt.sum()
loss.backward()

# Original call site: nt.share_memory_()
# Adapted call site: torch.nn.utils.clip_grad_norm_(nt, max_norm=1.0)
# This verifies that the similar API handles the NestedTensor structure without crashing.
try:
    total_norm = utils.clip_grad_norm_(nt, max_norm=1.0)
    # If the API is supported, it should return a float representing the total norm
    assert isinstance(total_norm, float)
except (NotImplementedError, TypeError) as e:
    # If the API is not supported for NestedTensor, it should raise a clear error,
    # not a segmentation fault (which was the original bug).
    print(f"API not supported for NestedTensor (expected behavior): {e}")

print("Test case executed successfully.")