import torch
import torch.nn as nn

class VarModel(nn.Module):
    def __init__(self):
        super(VarModel, self).__init__()

    def forward(self, x):
        return torch.var(x, correction=-1)

model = VarModel()
x = torch.ones([3])

# this succeeds
model(x)

# this fails
# ValueError: correction argument should be non-negative
torch.export.export(model, (x,))