import torch
import io
import sys

# Handle missing onnx dependency gracefully
try:
    import onnx
except ImportError:
    print("Skipping test: 'onnx' module is not installed.")
    sys.exit(0)

class SumModule(torch.nn.Module):
    def forward(self, x):
        return torch.sum(x, dim=1)

def test_onnx_dynamic_axes_with_backend_check():
    """
    Test case to verify that dynamic_axes in torch.onnx.export preserves
    custom axis names, leveraging backend availability checks.
    """
    # Leverage the similar API to check backend capabilities.
    # This is used here to determine the execution device, ensuring the test
    # respects the environment configuration.
    is_fa_available = torch.backends.cuda.is_flash_attention_available()
    
    device = 'cuda' if torch.cuda.is_available() and is_fa_available else 'cpu'
    
    model = SumModule().to(device)
    dummy_input = torch.ones(2, 2).to(device)
    
    # Use BytesIO for in-memory model export to avoid file I/O
    onnx_buffer = io.BytesIO()
    
    # Export the model with custom dynamic axis names
    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_buffer,
        input_names=["x"],
        output_names=["sum"],
        dynamic_axes={
            "x": {0: "my_custom_axis_name"},
            "sum": [0],
        },
    )
    
    # Load the exported ONNX model
    onnx_buffer.seek(0)
    onnx_model = onnx.load_from_string(onnx_buffer)
    
    # Verify the input dynamic axis name
    # Bug: In 2.9.0, this might return a serial number like "s77" instead of the custom name.
    input_value_info = onnx_model.graph.input[0]
    actual_dim_name = input_value_info.type.tensor_type.shape.dim[0].param_name
    
    expected_dim_name = "my_custom_axis_name"
    
    assert actual_dim_name == expected_dim_name, (
        f"Dynamic axis name mismatch for input 'x'. "
        f"Expected '{expected_dim_name}', but got '{actual_dim_name}'."
    )
    
    print("Test passed: Dynamic axis names are preserved correctly.")

if __name__ == "__main__":
    test_onnx_dynamic_axes_with_backend_check()