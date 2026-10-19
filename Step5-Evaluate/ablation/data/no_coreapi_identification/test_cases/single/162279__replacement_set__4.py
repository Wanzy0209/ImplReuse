import torch
import torch.nn as nn

class AnyDimsModelEmpty(nn.Module):
    def __init__(self):
        super(AnyDimsModelEmpty, self).__init__()

    def forward(self, x):
        # Using torch.ops.aten.any.dims with empty list
        y = torch.ops.aten.any.dims(x, [], False)
        return y

class AnyDimsModelNull(nn.Module):
    def __init__(self):
        super(AnyDimsModelNull, self).__init__()

    def forward(self, x):
        # Using torch.ops.aten.any.dims with None
        y = torch.ops.aten.any.dims(x, None, False)
        return y

def process(model, x):
    # Run eager mode
    model(x)
    # Run export
    exported_program = torch.export.export(model, (x,))
    return exported_program

if __name__ == "__main__":
    # Replacing torch.ones with torch.normal for input generation
    x = torch.normal(mean=0.0, std=1.0, size=(2, 3))

    model_empty = AnyDimsModelEmpty()
    model_null = AnyDimsModelNull()

    # Process the first model
    print("Processing AnyDimsModelEmpty...")
    ep_empty = process(model_empty, x)
    # Verify output shape (should be same as input for empty dims)
    assert ep_empty(x).shape == (2, 3), f"Test failed for AnyDimsModelEmpty: expected shape (2, 3), got {ep_empty(x).shape}"

    # Process the second model
    print("Processing AnyDimsModelNull...")
    ep_null = process(model_null, x)
    # Verify output shape (should be scalar for None dims)
    assert ep_null(x).shape == (), f"Test failed for AnyDimsModelNull: expected shape (), got {ep_null(x).shape}"

    print("All tests passed.")