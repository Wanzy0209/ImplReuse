# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
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

def process(model, x):
    print('model:', model)
    print('running eager mode...')
    model(x)
    print('exporting...')
    torch.export.export(model, (x,))
    print()