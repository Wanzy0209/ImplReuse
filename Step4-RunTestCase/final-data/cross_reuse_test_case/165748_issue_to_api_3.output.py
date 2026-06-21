import os
import collections
import sys

# Handle missing dependencies gracefully
try:
    import torch
    import onnx
except ImportError as e:
    print(f"Skipping test due to missing dependency: {e}")
    sys.exit(0)

# This test case reproduces the bug described in Issue 165748.
# It verifies that the custom names provided in the 'dynamic_axes' argument
# are preserved in the exported ONNX model and not replaced by serial numbers.

# To reflect the relationship with the similar API (tf.config.LogicalDevice),
# which is a structured configuration object, we define a configuration structure
# for the dynamic axes to ensure the test handles the configuration explicitly.

class DynamicAxisConfig(collections.namedtuple("DynamicAxisConfig", ["tensor_name", "dim_index", "dim_name"])):
    """Configuration for a single dynamic axis, mirroring structured config patterns."""
    pass

class SumModule(torch.nn.Module):
    def forward(self, x):
        return torch.sum(x, dim=1)

def test_dynamic_axes_preservation():
    # Setup configuration
    # The bug report indicates that "my_custom_axis_name" was being replaced by "s77".
    input_config = DynamicAxisConfig("x", 0, "my_custom_axis_name")
    output_config = DynamicAxisConfig("sum", 0, "sum_dynamic_dim") # Explicit name for output to check list vs dict behavior if needed, though bug focuses on input.
    
    # Construct the dynamic_axes dictionary as required by torch.onnx.export
    dynamic_axes = {
        input_config.tensor_name: {input_config.dim_index: input_config.dim_name},
        output_config.tensor_name: {output_config.dim_index: output_config.dim_name},
    }

    model_filename = "onnx_test.pb"

    try:
        # Export the model
        torch.onnx.export(
            SumModule(),
            (torch.ones(2, 2),),
            model_filename,
            input_names=[input_config.tensor_name],
            output_names=[output_config.tensor_name],
            dynamic_axes=dynamic_axes,
            opset_version=14, # Using a standard opset version
        )

        # Load and verify the ONNX model
        onnx_model = onnx.load(model_filename)
        
        # Check Input 'x'
        input_value_info = onnx_model.graph.input[0]
        # Access the dimension parameter for the first dimension (index 0)
        input_dim_param = input_value_info.type.tensor_type.shape.dim[0].dim_param
        
        # Assertion: The name should be the custom one, not a serial number like "s77"
        assert input_dim_param == input_config.dim_name, (
            f"Input dynamic axis name mismatch. Expected '{input_config.dim_name}', "
            f"but got '{input_dim_param}'. The bug (serial number generation) might be present."
        )
        
        # Check Output 'sum'
        output_value_info = onnx_model.graph.output[0]
        output_dim_param = output_value_info.type.tensor_type.shape.dim[0].dim_param
        
        assert output_dim_param == output_config.dim_name, (
            f"Output dynamic axis name mismatch. Expected '{output_config.dim_name}', "
            f"but got '{output_dim_param}'."
        )

        print("Test Passed: dynamic_axes names are correctly preserved.")

    finally:
        # Cleanup
        if os.path.exists(model_filename):
            os.remove(model_filename)

if __name__ == "__main__":
    test_dynamic_axes_preservation()