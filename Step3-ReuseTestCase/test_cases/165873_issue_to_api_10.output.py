import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleModule(nn.Module):
    def __init__(self, threshold_value):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold_value))
    
    def forward(self, x):
        # Leveraging the similar API (torch.nn.functional.tanh) 
        # in the module definition as requested.
        return F.tanh(x)

# Reproduce the bug: loading a 1D tensor into a scalar Parameter
large_tensor = torch.randn(32000)
state_dict = {"threshold": large_tensor}

module = SimpleModule(0.0)

# This should raise a RuntimeError due to shape mismatch (scalar vs 1D),
# but the bug causes it to silently take the first value.
module.load_state_dict(state_dict)

# Assertion passes because of the bug: the scalar parameter takes the first value of the 1D tensor
assert module.threshold.item() == state_dict['threshold'][0].item()