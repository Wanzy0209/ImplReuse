import torch
import torch.nn.functional as F

# Adapt the input structure from the original bug report.
# The original issue involved a 2D input tensor causing incorrect offset generation.
# Here we verify that the similar API, triplet_margin_with_distance_loss,
# correctly handles 2D inputs for anchor, positive, and negative tensors.

# Original input: torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=torch.long)
# We use float tensors for the loss function.
anchor = torch.tensor([[1.0, 2.0, 4.0, 5.0], [4.0, 3.0, 2.0, 9.0]], dtype=torch.float)
positive = torch.tensor([[0.5, 1.5, 3.5, 4.5], [3.5, 2.5, 1.5, 8.5]], dtype=torch.float)
negative = torch.tensor([[2.0, 3.0, 5.0, 6.0], [5.0, 4.0, 3.0, 10.0]], dtype=torch.float)

# Call the similar API
loss = F.triplet_margin_with_distance_loss(anchor, positive, negative)

# Assertions to verify correct behavior
assert loss.dim() == 0, "Loss output should be a scalar"
assert not torch.isnan(loss), "Loss output should not be NaN"
assert not torch.isinf(loss), "Loss output should not be Inf"

print("Test passed.")