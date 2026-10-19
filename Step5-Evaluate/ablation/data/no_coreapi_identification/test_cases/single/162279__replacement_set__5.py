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

if __name__ == "__main__":
    # Replacing torch.ones with torch.norm as requested
    # Original likely used: x = torch.ones(2, 3)
    # Using torch.norm to generate the input tensor
    x = torch.norm(torch.randn(2, 3))

    model_empty = AnyDimsModelEmpty()
    model_null = AnyDimsModelNull()

    # Process both models to reproduce the caching/shape issue
    process(model_empty, x)
    process(model_null, x)