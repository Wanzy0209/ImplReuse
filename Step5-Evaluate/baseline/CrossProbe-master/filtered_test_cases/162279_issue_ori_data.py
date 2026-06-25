import torch
import torch.nn as nn

class AnyDimsModelEmpty(nn.Module):
    def forward(self, x):
        return torch.ops.aten.any.dims(x, [], False)

class AnyDimsModelNull(nn.Module):
    def forward(self, x):
        return torch.ops.aten.any.dims(x, None, False)

model_empty = AnyDimsModelEmpty()
model_null = AnyDimsModelNull()
x = torch.ones([2, 3])

torch.export.export(model_empty, (x,))
torch.export.export(model_null, (x,))