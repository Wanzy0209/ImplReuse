import torch
import torch.onnx
import io

# This test case is based on Issue 166110: ConversionError with math.trunc.
# It leverages the code structure pattern of the similar API: tf.keras.initializers.HeNormal.
# 
# Similarity Analysis:
# - tf.keras.initializers.HeNormal defines a class to handle 'truncated_normal' distribution.
# - The bug involves 'math.trunc' failing during ONNX export.
# - We adapt the class structure of HeNormal to a PyTorch Module that performs truncation.

class TruncationModule(torch.nn.Module):
    """
    A PyTorch module that applies a truncation operation.
    Structurally similar to tf.keras.initializers.HeNormal (class-based configuration).
    """
    def __init__(self, scale=2.0):
        # Mirroring the __init__ pattern of HeNormal which accepts configuration parameters.
        super(TruncationModule, self).__init__()
        self.scale = scale

    def forward(self, x):
        # The bug report indicates a failure when translating 'math.trunc'.
        # We use torch.trunc here, which is the standard tensor operation for truncation.
        # In the context of the bug, this operation triggers the ConversionError.
        return torch.trunc(x * self.scale)

def test_onnx_export_with_trunc():
    """
    Reproduces the ONNX export error for models containing truncation operations.
    """
    # Initialize the model
    model = TruncationModule(scale=2.0)
    model.eval()

    # Prepare dummy inputs
    # Similar to the original bug report which used speech_feat and cache_source tensors.
    dummy_input = torch.randn(1, 10, dtype=torch.float32)

    # Attempt to export the model to ONNX
    # This is the specific API call (torch.onnx.export) that failed in the issue.
    onnx_buffer = io.BytesIO()
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_buffer,
            opset_version=17, # Using a standard opset version
            input_names=['input'],
            output_names=['output']
        )
        print("Export succeeded.")
        # Verify output
        assert onnx_buffer.getvalue() != b""
    except Exception as e:
        print(f"Export failed with error: {e}")
        # In a testing environment, we would assert the error type matches the bug report.
        raise

if __name__ == "__main__":
    test_onnx_export_with_trunc()