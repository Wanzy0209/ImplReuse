import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential

class AnyDimsModelEmpty(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Fix: Replace torch.ops.aten.any.dims with standard torch.any
        return torch.any(x, dim=[], keepdim=False)

class AnyDimsModelNull(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Fix: Replace torch.ops.aten.any.dims with standard torch.any
        return torch.any(x, dim=None, keepdim=False)

def process_with_checkpoint(model, x):
    # Adaptation: Replace torch.export.export with torch.utils.checkpoint.checkpoint_sequential
    # We wrap the model execution to fit the sequential functions requirement
    def run_model(inp):
        return model(inp)
    
    # Run the model through the checkpointing utility
    # segments=1 implies the whole list of functions is treated as one segment
    return checkpoint_sequential([run_model], 1, x)

if __name__ == "__main__":
    x = torch.randn(2, 3)
    
    model_empty = AnyDimsModelEmpty()
    model_null = AnyDimsModelNull()

    # Process first model
    print("Processing AnyDimsModelEmpty with checkpoint...")
    out_empty = process_with_checkpoint(model_empty, x)
    print("Output shape:", out_empty.shape)

    # Process second model
    print("Processing AnyDimsModelNull with checkpoint...")
    out_null = process_with_checkpoint(model_null, x)
    print("Output shape:", out_null.shape)

    # Verify correctness against eager execution to ensure no state pollution
    expected_empty = model_empty(x)
    expected_null = model_null(x)

    assert torch.equal(out_empty, expected_empty), "Output mismatch for AnyDimsModelEmpty"
    assert torch.equal(out_null, expected_null), "Output mismatch for AnyDimsModelNull"
    
    print("Test passed.")