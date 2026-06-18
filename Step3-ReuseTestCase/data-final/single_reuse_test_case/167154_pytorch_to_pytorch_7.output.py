import torch
from torch.nn import functional as F

# Setup non-contiguous tensors similar to the bug report
shape = (5, 499, 768)
stride = (0, 768, 1)
storage_offset = 0
numel = storage_offset + sum((shape[i] - 1) * stride[i] for i in range(len(shape))) + 1
base = torch.arange(numel, dtype=torch.float32, device="mps")

# Create inputs for triplet_margin_with_distance_loss
# Using the same strided logic that caused the buffer allocation error in F.linear
anchor = torch.as_strided(base, size=shape, stride=stride, storage_offset=storage_offset)
positive = torch.as_strided(base, size=shape, stride=stride, storage_offset=storage_offset)
negative = torch.as_strided(base, size=shape, stride=stride, storage_offset=storage_offset)

# Call the target API
# If the MPS buffer allocation regression affects this API, 
# this will raise an assertion error regarding buffer size.
F.triplet_margin_with_distance_loss(anchor, positive, negative)