import torch
import torch.onnx
import torch.nn as nn

# Define a model that mimics the conditional return logic found in the bug report.
# We leverage the similar API (torch.backends.mps.is_built) to drive the condition,
# creating a scenario where the output tuple might contain None.
class ModelWithConditionalNone(nn.Module):
    def forward(self, x):
        # Use torch.backends.mps.is_built to determine the output structure.
        # This mirrors the 'if not return_dict' logic in the original issue.
        if torch.backends.mps.is_built():
            # If MPS is built, return a valid tensor mask
            mask = torch.ones_like(x)
            return x, mask
        else:
            # If MPS is not built, return None as the second element.
            # This reproduces the bug: "Exporter crashes when fx node output includes None"
            return x, None

def test_onnx_export_with_none():
    """
    Test case to reproduce the ONNX exporter crash when a model returns 
    a tuple containing None.
    """
    model = ModelWithConditionalNone()
    model.eval()
    
    # Create a dummy input
    dummy_input = torch.randn(1, 3, 32, 32)
    
    # Attempt to export the model to ONNX
    # The bug occurs during the translation from the exported program to the ONNX graph
    # when the exporter encounters a None value in the node outputs.
    try:
        torch.onnx.export(
            model,
            dummy_input,
            "model_with_none_output.onnx",
            opset_version=17,
            input_names=["input"],
            output_names=["output", "mask"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        print("Test Passed: Export handled the None output gracefully.")
    except Exception as e:
        print(f"Test Failed: Exporter crashed with error: {e}")
        raise

if __name__ == "__main__":
    test_onnx_export_with_none()