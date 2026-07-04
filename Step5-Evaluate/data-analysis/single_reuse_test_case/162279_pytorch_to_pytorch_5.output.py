import torch
import torch.nn as nn

class AnyDimsModelEmpty(nn.Module):
    def __init__(self):
        super(AnyDimsModelEmpty, self).__init__()

    def forward(self, x):
        # Fix: Replace torch.ops.aten.any.dims with torch.any
        # An empty list of dimensions [] is equivalent to an empty tuple ()
        return torch.any(x, dim=(), keepdim=False)

class AnyDimsModelNull(nn.Module):
    def __init__(self):
        super(AnyDimsModelNull, self).__init__()

    def forward(self, x):
        # Fix: Replace torch.ops.aten.any.dims with torch.any
        # None for dimensions implies reducing over all dimensions
        return torch.any(x, keepdim=False)

def process(model, x):
    """
    Adapted process function to use torch.nn.DataParallel instead of torch.export.export.
    This verifies that DataParallel handles the models correctly without state pollution.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = x.to(device)
    
    # Original API: torch.export.export(model, (x,))
    # Similar API: Wrap in DataParallel and execute
    dp_model = nn.DataParallel(model)
    output = dp_model(x)
    return output

def test_dataparallel_any_dims():
    # Setup input
    x = torch.randn(2, 3)
    
    # Instantiate models
    model_empty = AnyDimsModelEmpty()
    model_null = AnyDimsModelNull()

    # Get expected outputs using eager mode (without DataParallel)
    # This establishes the ground truth for shapes and values
    expected_empty = model_empty(x)
    expected_null = model_null(x)

    # Process the first model
    output_empty = process(model_empty, x)
    
    # Process the second model sequentially
    # The bug report indicates that running export on the second model after the first
    # caused incorrect output shapes. We test if DataParallel is robust against this.
    output_null = process(model_null, x)

    # Verify shapes match the eager execution
    assert output_empty.shape == expected_empty.shape, \
        f"Shape mismatch for AnyDimsModelEmpty: DataParallel {output_empty.shape} vs Eager {expected_empty.shape}"
    
    assert output_null.shape == expected_null.shape, \
        f"Shape mismatch for AnyDimsModelNull: DataParallel {output_null.shape} vs Eager {expected_null.shape}"

    # Verify values match
    assert torch.equal(output_empty, expected_empty), "Value mismatch for AnyDimsModelEmpty"
    assert torch.equal(output_null, expected_null), "Value mismatch for AnyDimsModelNull"

    print("Test passed: torch.nn.DataParallel handles sequential models correctly.")

if __name__ == "__main__":
    test_dataparallel_any_dims()