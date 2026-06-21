import torch
import tempfile
import os
import sys

try:
    import onnx
except ImportError:
    print("Skipping test: 'onnx' module is not installed.")
    sys.exit(0)

# Reusing the pattern from tf.lookup.TextFileIndex to define axis indices.
# tf.lookup.TextFileIndex uses class attributes (e.g., WHOLE_LINE, LINE_NUMBER)
# to represent specific indices. We apply this pattern to define dynamic axes.
class AxisIndex:
    """
    Mimics the structure of tf.lookup.TextFileIndex to define 
    specific axis indices for dynamic axes configuration.
    """
    BATCH_DIM = 0
    SEQUENCE_DIM = 1

class SumModule(torch.nn.Module):
    def forward(self, x):
        return torch.sum(x, dim=1)

def test_dynamic_axes_preserves_custom_names():
    """
    Test that torch.onnx.export preserves custom dynamic axis names
    and does not replace them with serial numbers (e.g., 's77').
    """
    model = SumModule()
    dummy_input = torch.ones(2, 2)

    # Construct dynamic_axes using the AxisIndex class pattern
    # This mirrors how one might use TextFileIndex constants to specify file parsing logic.
    dynamic_axes_config = {
        "x": {AxisIndex.BATCH_DIM: "my_custom_batch_name"},
        "sum": [AxisIndex.BATCH_DIM],
    }

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=True) as f:
        # Export the model
        torch.onnx.export(
            model,
            dummy_input,
            f.name,
            input_names=["x"],
            output_names=["sum"],
            dynamic_axes=dynamic_axes_config,
        )

        # Load the exported model
        onnx_model = onnx.load(f.name)
        
        # Verify input axis name
        input_info = onnx_model.graph.input[0]
        input_dim_name = input_info.type.tensor_type.shape.dim[AxisIndex.BATCH_DIM].dim_param
        
        # Verify output axis name
        output_info = onnx_model.graph.output[0]
        output_dim_name = output_info.type.tensor_type.shape.dim[AxisIndex.BATCH_DIM].dim_param

        # The bug reported that names were replaced by serials like 's77'.
        # We assert that the custom name is preserved.
        assert input_dim_name == "my_custom_batch_name", \
            f"Input axis name mismatch: expected 'my_custom_batch_name', got '{input_dim_name}'"
        
        assert output_dim_name == "my_custom_batch_name", \
            f"Output axis name mismatch: expected 'my_custom_batch_name', got '{output_dim_name}'"

if __name__ == "__main__":
    test_dynamic_axes_preserves_custom_names()
    print("Test passed: Custom dynamic axis names are preserved.")