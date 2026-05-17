import torch
import torch.nn as nn

def test_torch_compile_isinf_dynamic_shapes():
    """
    Test case for Issue 161372: torch.compile regression in 2.8.0.
    
    The bug report indicates that torch.compile fails with a recompilation limit error
    and tensor size mismatch (expected 77, actual 78) when processing dynamic shapes.
    The traceback shows usage of torch.where inside torch._refs.
    This test leverages torch.isinf (Similar API) which is a pointwise operation
    often used with masking logic similar to the bug's context.
    """
    
    # Define a model that uses torch.isinf (the similar API)
    # and torch.where (as seen in the bug report traceback).
    class SimpleModel(nn.Module):
        def forward(self, x):
            # Use torch.isinf to create a mask
            mask = torch.isinf(x)
            
            # Use torch.where to apply the mask
            # This mirrors the pattern in the traceback: r = torch.where(mask, value, a)
            result = torch.where(mask, torch.zeros_like(x), x)
            return result

    model = SimpleModel()
    
    # Compile the model
    # In the bug report, this compilation failed or hit limits in PyTorch 2.8.0
    compiled_model = torch.compile(model)
    
    # Test Case 1: Sequence length 77
    # The bug report mentions 'expected 77'
    input_77 = torch.randn(1, 77, 128)
    input_77[0, 0, 0] = float('inf') # Ensure isinf is triggered
    
    try:
        output_77 = compiled_model(input_77)
        assert output_77.shape == (1, 77, 128)
        assert output_77[0, 0, 0] == 0.0 # Verify masking worked
    except Exception as e:
        print(f"Error during compilation or execution for size 77: {e}")
        raise

    # Test Case 2: Sequence length 78
    # The bug report mentions 'actual 78', causing the size mismatch error
    input_78 = torch.randn(1, 78, 128)
    input_78[0, 0, 0] = float('inf')
    
    try:
        output_78 = compiled_model(input_78)
        assert output_78.shape == (1, 78, 128)
        assert output_78[0, 0, 0] == 0.0
    except Exception as e:
        print(f"Error during compilation or execution for size 78: {e}")
        raise

    print("Test passed: torch.compile handles dynamic shapes with torch.isinf correctly.")

if __name__ == "__main__":
    test_torch_compile_isinf_dynamic_shapes()