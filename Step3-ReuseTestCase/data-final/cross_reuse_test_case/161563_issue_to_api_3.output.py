import torch
import torch.nn as nn

# Define a model that mimics the logic of tf.compat.v1.nn.pool
# The TF pool implementation involves explicit windowing and reduction.
# We implement a custom pooling layer to test the export behavior with such logic.
class CustomPoolModel(nn.Module):
    def forward(self, x):
        # Simulating the logic of tf.compat.v1.nn.pool
        # window_shape = [2, 2], strides = [2, 2]
        # Extract patches using unfold (similar to the windowing logic in TF)
        # x shape: (Batch, Channels, Height, Width)
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)
        
        # Perform reduction (Max Pooling)
        # This mimics the REDUCE operation in the TF docstring
        output, _ = patches.max(dim=-1).max(dim=-1)
        
        return output

def test_export_with_pool_like_logic():
    # Setup model and inputs
    model = CustomPoolModel()
    # Example inputs: Batch=1, Channels=3, Height=10, Width=10
    example_inputs = (torch.randn(1, 3, 10, 10),)

    # Attempt to export the model
    # This mirrors the original bug report's usage of torch.export.export
    try:
        ep = torch.export.export(model, example_inputs)
        # If successful, verify the exported program
        assert ep is not None
        print("Export successful.")
    except AssertionError as e:
        # Check for the specific error mentioned in the bug report
        if "Current active mode" in str(e):
            print(f"Bug reproduced: {e}")
            # Re-raise to indicate test failure (or handle as expected for bug reproduction)
            raise
        else:
            raise

if __name__ == "__main__":
    test_export_with_pool_like_logic()