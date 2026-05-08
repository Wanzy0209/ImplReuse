import torch
import torch.nn as nn

class CDistModel(nn.Module):
    def __init__(self):
        super(CDistModel, self).__init__()

    def forward(self, x, y, compute_mode):
        return torch.ops.aten._cdist_forward(x, y, p=2.0, compute_mode=compute_mode)

x = torch.ones([3, 3])
y = torch.ones([3, 3])

model = CDistModel()

# this succeeds
model(x, y, None)
model(x, y, 0)
model(x, y, 1)
model(x, y, 2)

# this succeeds
torch.export.export(model, (x, y, None))
torch.export.export(model, (x, y, 1))
torch.export.export(model, (x, y, 2))

# this fails
# RuntimeError: possible modes: None, 1, 2, but was: 0
torch.export.export(model, (x, y, 0))

# Notice that if we force a failure in the aten op, by passing a wrong
# compute_mode, we get a different error message:
# RuntimeError: possible modes: 0, 1, 2, but was: 3
model(x, y, 3)