import torch
import torch.nn as nn
import torch.export

class AnyDimsModelEmpty(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Using empty list for dims
        return torch.ops.aten.any.dims(x, [], False)

class AnyDimsModelNull(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Using None for dims
        return torch.ops.aten.any.dims(x, None, False)

def test_export_shape_correctness():
    # Setup input using torch.linspace as requested (replacing torch.ones)
    # Creating a 2x3 tensor
    x = torch.linspace(0, 1, steps=6).reshape(2, 3)

    model_empty = AnyDimsModelEmpty()
    model_null = AnyDimsModelNull()

    # Process first model (Empty)
    print("Processing AnyDimsModelEmpty...")
    exported_empty = torch.export.export(model_empty, (x,))
    out_empty = exported_empty(x)
    # Expected shape: (2, 3) because dims=[] means no reduction
    assert out_empty.shape == (2, 3), f"Empty model failed: expected (2, 3), got {out_empty.shape}"

    # Process second model (Null)
    # The bug report indicates that processing a second model might result in incorrect shape
    # due to caching issues.
    print("Processing AnyDimsModelNull...")
    exported_null = torch.export.export(model_null, (x,))
    out_null = exported_null(x)
    # Expected shape: torch.Size([]) because dims=None means reduce all
    assert out_null.shape == torch.Size([]), f"Null model failed: expected [], got {out_null.shape}"

    # Verify order independence by processing in reverse order
    print("Processing AnyDimsModelNull (2nd run)...")
    model_null_2 = AnyDimsModelNull()
    exported_null_2 = torch.export.export(model_null_2, (x,))
    out_null_2 = exported_null_2(x)
    assert out_null_2.shape == torch.Size([]), f"Null model (2nd run) failed: expected [], got {out_null_2.shape}"

    print("Processing AnyDimsModelEmpty (2nd run)...")
    model_empty_2 = AnyDimsModelEmpty()
    exported_empty_2 = torch.export.export(model_empty_2, (x,))
    out_empty_2 = exported_empty_2(x)
    assert out_empty_2.shape == (2, 3), f"Empty model (2nd run) failed: expected (2, 3), got {out_empty_2.shape}"

    print("All tests passed.")

if __name__ == "__main__":
    test_export_shape_correctness()