import torch
from torch import nn

class LazyBilinear(nn.Bilinear):
    def __init__(self, out_features: int, bias: bool = True):
        super().__init__(0, 0, out_features, bias=bias)

# This will raise ValueError due to the check
try:
    LazyBilinear(10)
except ValueError as e:
    print(f"Error: {e}")