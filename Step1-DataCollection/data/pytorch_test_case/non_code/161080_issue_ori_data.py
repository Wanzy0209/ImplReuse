import torch
import torch.nn as nn

class ExpandModel(nn.Module):
    def __init__(self):
        super(ExpandModel, self).__init__()

    def forward(self, x, implicit):
        return torch.expand_copy(x, [3, 3], implicit=implicit)

model = ExpandModel()
x = torch.ones([3])

# this succeeds
model(x, False)
model(x, True)
torch.export.export(model, (x, False))

# this fails
# TypeError: expand() got an unexpected keyword argument 'implicit'
torch.export.export(model, (x, True))