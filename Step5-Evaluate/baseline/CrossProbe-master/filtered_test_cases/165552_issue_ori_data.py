import torch
import torch.nn as nn

class ReLUTracker(nn.Module):
    def __init__(self):
        super().__init__()
        self.dead_count = 0
        self.total = 0
    
    def forward(self, x):
        out = torch.relu(x)
        self.dead_count += (out == 0).sum().item()
        self.total += out.numel()
        return out

# Usage
layer = ReLUTracker()
x = torch.randn(10, 5)
y = layer(x)
print(f'Dead ratio: {layer.dead_count/layer.total:.2%}')