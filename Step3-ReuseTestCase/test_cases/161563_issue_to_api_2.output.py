import torch
import torch.nn as nn

# Define a minimal model that performs a diagonal extraction operation.
# This leverages the semantic pattern of the similar API (tf.linalg.tensor_diag_part)
# within the context of the original bug reproduction logic (torch.export.export).
class DiagonalModel(nn.Module):
    def forward(self, x):
        # torch.diagonal is the PyTorch equivalent of tf.linalg.tensor_diag_part
        return torch.diagonal(x)

def test_export_with_diagonal_op():
    # Instantiate the model
    model = DiagonalModel()
    
    # Create example inputs. 
    # Using a 3D tensor to mimic batched inputs often found in models like Gemma.
    example_inputs = (torch.randn(2, 4, 4),)

    # Reproduce the original bug logic: calling torch.export.export
    # The bug report indicates an AssertionError related to "Current active mode not registered"
    # when exporting models involving specific tensor operations (like vmap/diag).
    try:
        ep = torch.export.export(model, example_inputs)
        
        # Verify the exported program runs correctly
        output = ep.module(*example_inputs)
        
        # Basic assertion to ensure functionality is preserved
        expected_shape = torch.Size([2, 4])
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        
        print("Test Passed: Export successful with diagonal operation.")
        
    except AssertionError as e:
        # Check if we hit the specific error mentioned in the bug report
        if "Current active mode" in str(e):
            print(f"Bug Reproduced: {e}")
            raise
        else:
            # Re-raise if it's a different assertion error
            raise

if __name__ == "__main__":
    test_export_with_diagonal_op()