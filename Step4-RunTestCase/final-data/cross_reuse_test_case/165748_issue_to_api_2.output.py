import sys
import os

# Handle missing dependencies by skipping the test
try:
    import torch
    import onnx
except ImportError as e:
    print(f"Skipping test: Missing dependency - {e}")
    sys.exit(0)

def test_dynamic_axes_custom_names():
    """
    Test case to verify that torch.onnx.export preserves custom dynamic axis names.
    This addresses the issue where custom names were replaced by serial numbers (e.g., 's77').
    
    The test logic mirrors the pattern of checking a specific feature state,
    similar to how one might check if a resource variable mode is enabled.
    """
    
    # Define a simple module
    class SumModule(torch.nn.Module):
        def forward(self, x):
            return torch.sum(x, dim=1)

    model = SumModule()
    dummy_input = torch.ones(2, 2)
    onnx_path = "test_dynamic_axes.onnx"
    
    # Define the expected custom names
    expected_input_name = "my_custom_batch_size"
    expected_output_name = "my_custom_output_dim"

    try:
        # Export the model with dynamic_axes
        torch.onnx.export(
            model,
            (dummy_input,),
            onnx_path,
            input_names=["x"],
            output_names=["sum"],
            dynamic_axes={
                "x": {0: expected_input_name},
                "sum": {0: expected_output_name},
            },
        )

        # Load the exported ONNX model
        onnx_model = onnx.load(onnx_path)

        # Verify Input Axis Name
        # We check the graph input to ensure the custom name is preserved
        input_info = onnx_model.graph.input[0]
        actual_input_dim_name = input_info.type.tensor_type.shape.dim[0].dim_param
        
        assert actual_input_dim_name == expected_input_name, (
            f"Input dynamic axis name mismatch. "
            f"Expected '{expected_input_name}', but got '{actual_input_dim_name}'. "
            f"This indicates the bug where names are replaced by serial numbers."
        )

        # Verify Output Axis Name
        output_info = onnx_model.graph.output[0]
        actual_output_dim_name = output_info.type.tensor_type.shape.dim[0].dim_param
        
        assert actual_output_dim_name == expected_output_name, (
            f"Output dynamic axis name mismatch. "
            f"Expected '{expected_output_name}', but got '{actual_output_dim_name}'."
        )

        print("Test Passed: Dynamic axis names are correctly preserved.")

    finally:
        # Clean up the generated file
        if os.path.exists(onnx_path):
            os.remove(onnx_path)

if __name__ == "__main__":
    test_dynamic_axes_custom_names()