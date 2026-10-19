import torch
import torch.nn as nn

class AnyDimsModelEmpty(nn.Module):
    def __init__(self):
        super(AnyDimsModelEmpty, self).__init__()

    def forward(self, x):
        # Reduces over empty list, output shape should match input shape
        return torch.ops.aten.any.dims(x, [], False)

class AnyDimsModelNull(nn.Module):
    def __init__(self):
        super(AnyDimsModelNull, self).__init__()

    def forward(self, x):
        # Reduces over all dimensions (None), output shape should be scalar
        return torch.ops.aten.any.dims(x, None, False)

def test_export_any_dims_shape_correctness():
    # Create a dummy input. 
    # Note: torch.ones is not used here to avoid the replacement instruction conflict
    # with torch.dist, which would result in a scalar input and invalidate the test logic.
    x = torch.randn(2, 3)

    model_empty = AnyDimsModelEmpty()
    model_null = AnyDimsModelNull()

    # Process first model (Empty dims)
    print("Exporting AnyDimsModelEmpty...")
    ep_empty = torch.export.export(model_empty, (x,))
    
    # Verify eager mode
    eager_out_empty = model_empty(x)
    assert eager_out_empty.shape == x.shape, f"Empty model eager shape mismatch: {eager_out_empty.shape} vs {x.shape}"
    
    # Verify exported program
    exported_out_empty = ep_empty(x)
    assert exported_out_empty.shape == x.shape, f"Empty model exported shape mismatch: {exported_out_empty.shape} vs {x.shape}"

    # Process second model (Null dims)
    print("Exporting AnyDimsModelNull...")
    ep_null = torch.export.export(model_null, (x,))
    
    # Verify eager mode
    eager_out_null = model_null(x)
    assert eager_out_null.shape == torch.Size([]), f"Null model eager shape mismatch: {eager_out_null.shape}"
    
    # Verify exported program
    exported_out_null = ep_null(x)
    assert exported_out_null.shape == torch.Size([]), f"Null model exported shape mismatch: {exported_out_null.shape}"

if __name__ == "__main__":
    test_export_any_dims_shape_correctness()