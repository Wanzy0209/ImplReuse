import torch
import torch.nn as nn

class SimpleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(0.0))

large_tensor = torch.randn(32000)
state_dict = {"threshold": large_tensor}

module = SimpleModule()
module.load_state_dict(state_dict)  # Should raise error but doesn't
assert module.threshold.item() == state_dict['threshold'][0].item()