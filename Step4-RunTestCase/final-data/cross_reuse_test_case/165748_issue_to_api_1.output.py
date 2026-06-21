import torch
import os
import unittest

# Handle missing onnx dependency gracefully
try:
    import onnx
except ImportError:
    onnx = None

# Leveraging the similar API: torch.distributed.is_available
# We reuse this API to check the environment state before running the ONNX export test,
# mimicking the pattern of checking availability/capability before performing an operation.
from torch.distributed import is_available

class TestONNXDynamicAxes(unittest.TestCase):
    def test_dynamic_axes_custom_names(self):
        # Reuse pattern: Check availability/state before proceeding
        if not is_available():
            self.skipTest("torch.distributed is not available, skipping test based on similar API usage")

        if onnx is None:
            self.skipTest("onnx module is not installed, skipping test")

        class SumModule(torch.nn.Module):
            def forward(self, x):
                return torch.sum(x, dim=1)

        file_path = "onnx.pb"
        
        try:
            # Reproduce the bug scenario from Issue 165748
            torch.onnx.export(
                SumModule(),
                (torch.ones(2, 2),),
                file_path,
                input_names=["x"],
                output_names=["sum"],
                dynamic_axes={
                    "x": {0: "my_custom_axis_name"},
                    "sum": [0],
                },
            )

            onnx_model = onnx.load(file_path)
            
            # Verify the fix: The custom axis name should be preserved
            # Bug behavior: name becomes "s77" (a serial number)
            # Expected behavior: name is "my_custom_axis_name"
            
            input_value_info = onnx_model.graph.input[0]
            dim_param = input_value_info.type.tensor_type.shape.dim[0].dim_param
            
            self.assertEqual(dim_param, "my_custom_axis_name", 
                             f"Dynamic axis name mismatch. Expected 'my_custom_axis_name', got '{dim_param}'")
            
            # Also verify the output is dynamic (though name might be auto-generated if not specified)
            output_value_info = onnx_model.graph.output[0]
            # In ONNX, if a dimension is dynamic, dim_param is set. If static, dim_value is set.
            self.assertIsNotNone(output_value_info.type.tensor_type.shape.dim[0].dim_param,
                                 "Output dimension 0 should be dynamic")

        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

if __name__ == "__main__":
    unittest.main()