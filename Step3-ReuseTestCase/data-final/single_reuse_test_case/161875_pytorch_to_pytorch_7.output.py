import torch
import torch.nn.functional as F

# Adapt input to 4D (Batch, Channel, Height, Width) as required by upsample_bilinear
input_data = torch.randn(1, 3, 32, 32)

# Use the extreme value from the original bug report for the size parameter
# This attempts to trigger a similar segmentation fault or memory error
output = F.upsample_bilinear(input_data, size=9223372036854775803)