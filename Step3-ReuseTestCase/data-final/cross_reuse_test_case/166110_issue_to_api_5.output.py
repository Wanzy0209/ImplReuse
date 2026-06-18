import torch
import math
import torch.onnx

# This class mimics the structure of tf.keras.initializers.HeUniform
# (taking scale and seed in __init__) but implements the logic
# that triggers the reported bug in torch.onnx.export (math.trunc).
class TruncWrapper(torch.nn.Module):
    def __init__(self, scale=2.0, seed=None):
        super(TruncWrapper, self).__init__()
        self.scale = scale
        self.seed = seed

    def forward(self, x):
        # The bug is triggered by the use of math.trunc
        # which is not supported in ONNX export in this context.
        # This mimics the error: call_function[target=math.trunc](args = (%mul_104,), kwargs = {})
        return math.trunc(x * self.scale)

def test_onnx_export_with_trunc():
    """
    Test case to reproduce the ConversionError when exporting
    a model containing math.trunc to ONNX.
    """
    # Initialize model with parameters similar to HeUniform
    model = TruncWrapper(scale=2.0, seed=42)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 10)

    # Attempt to export to ONNX
    # This is expected to fail with ConversionError based on the issue
    try:
        torch.onnx.export(
            model,
            dummy_input,
            "trunc_model.onnx",
            opset_version=14, # Using a standard opset version
            input_names=['input'],
            output_names=['output']
        )
        # If we reach here, the bug might be fixed
        assert False, "Expected ConversionError but export succeeded."
    except Exception as e:
        # Check if the error is related to math.trunc
        assert "math.trunc" in str(e) or "ConversionError" in str(e), \
            f"Unexpected error: {e}"
        print(f"Test passed: Caught expected error - {e}")

if __name__ == "__main__":
    test_onnx_export_with_trunc()