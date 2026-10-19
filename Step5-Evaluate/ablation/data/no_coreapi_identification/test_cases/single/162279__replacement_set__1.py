import torch
import torch.nn as nn

class AnyDimsModelEmpty(nn.Module):
    def __init__(self):
        super(AnyDimsModelEmpty, self).__init__()

    def forward(self, x):
        print('input shape:', x.shape)
        y = torch.ops.aten.any.dims(x, [], False)
        print('output shape:', y.shape)
        return y

class AnyDimsModelNull(nn.Module):
    def __init__(self):
        super(AnyDimsModelNull, self).__init__()

    def forward(self, x):
        print('input shape:', x.shape)
        y = torch.ops.aten.any.dims(x, None, False)
        print('output shape:', y.shape)
        return y

# Setup input using torch.zeros (replacement for torch.ones)
x = torch.zeros(2, 3, dtype=torch.bool)

# Instantiate models
model_empty = AnyDimsModelEmpty()
model_null = AnyDimsModelNull()

# Process first model
print('model:', model_empty)
print('running eager mode...')
model_empty(x)
print('exporting...')
torch.export.export(model_empty, (x,))
print()

# Process second model
print('model:', model_null)
print('running eager mode...')
model_null(x)
print('exporting...')
torch.export.export(model_null, (x,))
print()