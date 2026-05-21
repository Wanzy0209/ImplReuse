import torch
import onnx
import os

def test_onnx_dynamic_axes_bug():
    """
    Test case to verify that torch.onnx.export preserves custom dynamic_axes names
    and does not generate serial numbers like 's77' (Issue 165748).
    Leverages torch.backends.opt_einsum.is_available to check environment status.
    """
    
    # Leverage the similar API: Check backend availability
    # This pattern is reused here to demonstrate environment checking before export.
    if not torch.backends.opt_einsum.is_available():
        print("Warning: opt_einsum is not available. Proceeding with ONNX export test.")

    class SumModule(torch.nn.Module):
        def forward(self, x):
            return torch.sum(x, dim=1)

    model = SumModule()
    dummy_input = torch.ones(2, 2)
    export_path = "onnx_dynamic_axes_test.pb"
    
    # Custom names expected in the exported model
    custom_input_axis_name = "my_custom_axis_name"
    custom_output_axis_name = "my_custom_output_name"

    try:
        # Reproduce the bug scenario
        torch.onnx.export(
            model,
            (dummy_input,),
            export_path,
            input_names=["x"],
            output_names=["sum"],
            dynamic_axes={
                "x": {0: custom_input_axis_name},
                "sum": {0: custom_output_axis_name},
            },
        )

        # Load the exported model to inspect the graph
        onnx_model = onnx.load(export_path)

        # Verify Input Dynamic Axis Name
        input_info = onnx_model.graph.input[0]
        input_dim_name = input_info.type.tensor_type.shape.dim[0].dim_param
        
        # Verify Output Dynamic Axis Name
        output_info = onnx_model.graph.output[0]
        output_dim_name = output_info.type.tensor_type.shape.dim[0].dim_param

        # Assertions to check if the bug is present
        # The bug report indicates names become "s77". We expect the custom names.
        assert input_dim_name == custom_input_axis_name, \
            f"Bug reproduced: Input axis name is '{input_dim_name}', expected '{custom_input_axis_name}'"
        
        assert output_dim_name == custom_output_axis_name, \
            f"Bug reproduced: Output axis name is '{output_dim_name}', expected '{custom_output_axis_name}'"

        print("Test Passed: Dynamic axes names are preserved correctly.")

    finally:
        # Clean up the generated file
        if os.path.exists(export_path):
            os.remove(export_path)

if __name__ == "__main__":
    test_onnx_dynamic_axes_bug()