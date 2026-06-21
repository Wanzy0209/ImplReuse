import unittest
import torch
import torch.nn as nn
import os
import numpy as np

# Handle missing onnxruntime dependency gracefully
try:
    import onnxruntime as ort
except ImportError:
    ort = None

@unittest.skipIf(ort is None, "onnxruntime is not installed")
class TestAtan2ONNXExport(unittest.TestCase):
    """
    Test case to verify that torch.atan2(0, 0) exports correctly to ONNX.
    
    Bug Description:
    torch.atan2(x, y) returns zero when both inputs are zero. 
    The ONNX decomposition of atan2 produces NaN in this case.
    """
    
    def setUp(self):
        self.onnx_model_path = "atan2_test.onnx"

    def tearDown(self):
        if os.path.exists(self.onnx_model_path):
            os.remove(self.onnx_model_path)

    def test_atan2_zero_input_consistency(self):
        class Net(nn.Module):
            def forward(self, x, y):
                return torch.atan2(x, y)

        net = Net()
        # Input: x=0, y=0
        x = torch.tensor([0.0])
        y = torch.tensor([0.0])

        # 1. Get PyTorch result
        torch_result = net(x, y)
        
        # Expected behavior in PyTorch: atan2(0, 0) = 0
        expected_torch_result = torch.tensor([0.0])
        self.assertTrue(
            torch.allclose(torch_result, expected_torch_result), 
            f"PyTorch atan2(0, 0) should be 0, got {torch_result.item()}"
        )

        # 2. Export to ONNX
        try:
            torch.onnx.export(
                net, 
                (x, y), 
                self.onnx_model_path, 
                input_names=["x", "y"], 
                output_names=["output"],
                opset_version=14 # Using a standard opset version
            )
        except Exception as e:
            self.fail(f"ONNX export failed: {e}")

        # 3. Get ONNX Runtime result
        try:
            sess = ort.InferenceSession(self.onnx_model_path)
            ort_result = sess.run(["output"], {"x": x.numpy(), "y": y.numpy()})[0]
        except Exception as e:
            self.fail(f"ONNX Runtime inference failed: {e}")

        # 4. Verify consistency
        # The bug manifests here: ort_result might be NaN while torch_result is 0
        # We check if they are close. If ort_result is NaN, this will fail.
        self.assertTrue(
            np.allclose(torch_result.numpy(), ort_result, equal_nan=True),
            f"ONNX output ({ort_result[0]}) does not match PyTorch output ({torch_result.item()}) for atan2(0, 0)"
        )

if __name__ == "__main__":
    unittest.main()