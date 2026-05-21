import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential
from torch.export import export, Dim

def test_checkpoint_sequential_export_dynamic_shapes():
    """
    Test case to verify torch.utils.checkpoint.checkpoint_sequential 
    works correctly with torch.export when using dynamic shapes (sequence length).
    
    This is adapted from the bug report where torch.export failed with 
    'Constraints violated (seq)' on models with dynamic sequence lengths.
    """
    
    # 1. Setup a simple sequential model to be checkpointed
    # We use multiple layers to make the segmentation meaningful
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(10, 10),
                nn.ReLU(),
                nn.Linear(10, 10),
                nn.ReLU(),
                nn.Linear(10, 10)
            )
        
        def forward(self, x):
            return self.layers(x)

    model = SimpleModel()

    # 2. Define the function using checkpoint_sequential
    # We wrap the model execution in checkpoint_sequential
    def run_with_checkpoint(x):
        # Split the model into 2 segments. 
        # use_reentrant=False is recommended in newer PyTorch versions.
        return checkpoint_sequential(model.layers, 2, x, use_reentrant=False)

    # 3. Define dynamic shape constraints
    # The bug report specifically mentions issues with 'seq' constraints.
    # We define a dynamic dimension for the sequence length (dim 1).
    batch_dim = Dim("batch", min=1, max=4)
    seq_dim = Dim("seq", min=10, max=512) # Constraint range similar to bug report context

    # Example input with a specific sequence length
    example_args = (torch.randn(2, 50, 10),)
    dynamic_shapes = ({0: batch_dim, 1: seq_dim},)

    # 4. Attempt to export the function
    # This is where the original bug (UserError: Constraints violated) occurred.
    try:
        exported_program = export(
            run_with_checkpoint, 
            args=example_args, 
            dynamic_shapes=dynamic_shapes
        )
    except Exception as e:
        print(f"Export failed: {e}")
        raise

    # 5. Verify the exported program with different sequence lengths
    # to ensure constraints are not violated during execution.
    
    # Test with a smaller sequence length (within min constraint)
    input_small = torch.randn(2, 20, 10)
    output_small = exported_program(input_small)
    assert output_small.shape == (2, 20, 10), "Output shape mismatch for small sequence"

    # Test with a larger sequence length (within max constraint)
    input_large = torch.randn(2, 100, 10)
    output_large = exported_program(input_large)
    assert output_large.shape == (2, 100, 10), "Output shape mismatch for large sequence"

    # Test with max constraint boundary
    input_max = torch.randn(2, 512, 10)
    output_max = exported_program(input_max)
    assert output_max.shape == (2, 512, 10), "Output shape mismatch for max sequence"

    print("Test passed: checkpoint_sequential handles dynamic shapes in torch.export correctly.")

if __name__ == "__main__":
    test_checkpoint_sequential_export_dynamic_shapes()