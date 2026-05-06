import torch
import torch.nn as nn

class AddModel(nn.Module):
    def __init__(self):
        super(AddModel, self).__init__()

    def forward(self, x):
        return torch.ops.aten.add.Scalar(x, 1, 2)

x = torch.ones([3])

model = AddModel()

# this succeeds
output = model(x)

# this fails
# TypeError: too many positional arguments
torch.export.export(model, (x,))