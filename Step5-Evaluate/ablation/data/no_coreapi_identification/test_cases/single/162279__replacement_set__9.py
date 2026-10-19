import torch
import torch.nn as nn

class AnyDimsModelEmpty(nn.Module):
    def __init__(self):
        super(AnyDimsModelEmpty, self).__init__()

    def forward(self, x):
        # print('input shape:', x.shape)
        y = torch.ops.aten.any.dims(x, [], False)
        # print('output shape:', y.shape)
        return y

class AnyDimsModelNull(nn.Module):
    def __init__(self):
        super(AnyDimsModelNull, self).__init__()

    def forward(self, x):
        # print('input shape:', x.shape)
        y = torch.ops.aten.any.dims(x, None, False)
        # print('output shape:', y.shape)
        return y

def process(model, x):
    # print('model:', model)
    # print('running eager mode...')
    model(x)
    # print('exporting...')
    exported_program = torch.export.export(model, (x,))
    # print()
    return exported_program

if __name__ == "__main__":
    # Setup input tensor.
    # Original API: torch.ones
    # Replacement API: torch.matrix_power
    # torch.matrix_power requires a square matrix and an exponent.
    # We use torch.eye(3) to create a square matrix and power 1.
    x = torch.matrix_power(torch.eye(3), 1)

    model_empty = AnyDimsModelEmpty()
    model_null = AnyDimsModelNull()

    # Process the first model
    print("Processing AnyDimsModelEmpty...")
    ep_empty = process(model_empty, x)
    print(f"Exported output shape (Empty): {ep_empty.output_schema[0].shape}")

    # Process the second model
    print("Processing AnyDimsModelNull...")
    ep_null = process(model_null, x)
    print(f"Exported output shape (Null): {ep_null.output_schema[0].shape}")

    # Verify shapes are correct (both should be scalar)
    # The bug report implies the second one might be wrong.
    assert ep_empty.output_schema[0].shape == torch.Size([]), "Shape mismatch for AnyDimsModelEmpty"
    assert ep_null.output_schema[0].shape == torch.Size([]), "Shape mismatch for AnyDimsModelNull"