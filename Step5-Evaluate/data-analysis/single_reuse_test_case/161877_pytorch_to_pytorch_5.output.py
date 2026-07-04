import torch
from torch.nn import GroupNorm

input_data = torch.randn(1, 16, 100)

class MyGroupNormModule(torch.nn.Module):
    def __init__(self):
        super(MyGroupNormModule, self).__init__()
        # Fix: num_groups must be a divisor of num_channels (16).
        # Using 16 (the maximum valid number of groups) to test boundary conditions.
        self.norm = GroupNorm(num_groups=16, num_channels=16)
        self.add_module(name='norm', module=self.norm)

model = MyGroupNormModule()
output = model.norm(input_data)