import torch
import torch.nn as nn


class SimpleModule(nn.Module):
    def __init__(self, threshold_value):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold_value))
    
    def forward(self, x):
        return x

large_tensor = torch.randn(32000)
state_dict = {"threshold": large_tensor}

module = SimpleModule(0.0)
module.load_state_dict(state_dict)
assert module.threshold.item() == state_dict['threshold'][0].item()