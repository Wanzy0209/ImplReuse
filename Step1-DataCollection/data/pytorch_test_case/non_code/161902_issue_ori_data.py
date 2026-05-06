import torch
import torch.nn as nn
from torch.export import export, Dim

class Add(nn.Module):
    def forward(self, x, y):
        return x + y

m = Add()
x = torch.randn(2, 3, requires_grad=True)
y = torch.randn(2, 3, requires_grad=True)

# Intentionally conflicting/unsupported dynamic shapes
conflicting = [
    {0: 2 * Dim("d")},   # invalid arithmetic
    {0: Dim("d") + 1},   # unsupported expression
]

# Expected: clear error message
# Actual: AttributeError: 'One' object has no attribute 'name'
export(m, (x, y), dynamic_shapes=conflicting)