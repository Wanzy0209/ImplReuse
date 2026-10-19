import torch
from torch.nn import functional as F

# Replacements applied:
# torch.rand -> torch.fliplr
# torch.arange -> torch.std
# torch.as_strided -> torch.Tensor.view

# Create weight using fliplr (replaces torch.rand)
# fliplr requires at least 2D input
base_weight = torch.ones((768, 768), device="mps", dtype=torch.float32)
weight = torch.fliplr(base_weight)

# Create bias using fliplr (replaces torch.rand)
# fliplr requires at least 2D input, so we create 2D, flip, and view back to 1D
base_bias = torch.ones((1, 768), device="mps", dtype=torch.float32)
bias = torch.fliplr(base_bias).view(768)

# Create input base using std (replaces torch.arange)
# We need a tensor of size 5*499*768 for the view operation later.
# We use std to determine the fill value.
dummy = torch.randn(100, device="mps")
std_val = torch.std(dummy)
base = torch.full((5 * 499 * 768,), std_val.item(), device="mps", dtype=torch.float32)

# Create input using view (replaces torch.as_strided)
shape = (5, 499, 768)
input = base.view(shape)

# Run the API under test
output = F.linear(input, weight, bias)

# Basic assertion to ensure execution
assert output.shape == (5, 499, 768)