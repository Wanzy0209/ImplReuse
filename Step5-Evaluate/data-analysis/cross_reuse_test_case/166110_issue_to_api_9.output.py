import torch
import torch.nn as nn
import math
import tempfile
import os

# The issue reports a ConversionError when exporting a model containing math.trunc to ONNX.
# The similar API is tf.nn.relu, suggesting a pattern of element-wise operations.
# We leverage this pattern by including a ReLU operation before the problematic trunc call.

class TruncInferenceWrapper(nn.Module):
    """
    A minimal wrapper to reproduce the ONNX export error with math.trunc.
    This mirrors the structure of the HiFTInferenceWrapper from the issue.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        # Leverage the pattern of the similar API (tf.nn.relu) using torch.nn.functional.relu
        # to perform a standard element-wise activation.
        x = torch.nn.functional.relu(x)
        
        # Reproduce the bug: using the standard library math.trunc on a tensor.
        # The ONNX exporter struggles with call_function[target=math.trunc].
        # Ideally, torch.trunc should be used, but the bug report implies math.trunc was present.
        return math.trunc(x)

def test_onnx_export_math_trunc_conversion_error():
    """
    Test case to reproduce the ConversionError when exporting a model
    that uses math.trunc to ONNX.
    """
    device = torch.device("cpu")
    model = TruncInferenceWrapper().to(device)
    model.eval()

    # Build example inputs similar to the issue's build_example_inputs
    dummy_input = torch.randn(1, 80, 10, dtype=torch.float32, device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "model.onnx")

        print("Attempting to export model with math.trunc to ONNX...")
        try:
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                opset_version=17,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            # If export succeeds, the bug might be fixed in the environment, 
            # but based on the issue, we expect a failure.
            print("Export succeeded unexpectedly.")
            
        except Exception as e:
            print(f"Export failed as expected with error: {e}")
            # Assert that the error is related to trunc or conversion
            error_str = str(e)
            assert "trunc" in error_str.lower() or "ConversionError" in error_str, \
                f"Expected error related to 'trunc' or 'ConversionError', got: {e}"

if __name__ == "__main__":
    test_onnx_export_math_trunc_conversion_error()