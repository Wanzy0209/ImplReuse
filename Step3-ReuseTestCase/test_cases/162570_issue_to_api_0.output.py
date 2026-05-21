import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np
import tempfile
import os

def test_atan2_zero_input_consistency():
    """
    Test that torch.atan2(0, 0) returns 0 and that the ONNX export
    produces the same result (0) rather than NaN.
    
    Related to Issue ID: 162570
    """
    
    class Atan2Net(nn.Module):
        def forward(self, x, y):
            return torch.atan2(x, y)

    # 1. Test Native PyTorch Behavior
    net = Atan2Net()
    x = torch.tensor([0.0])
    y = torch.tensor([0.0])
    
    torch_result = net(x, y)
    
    # Assert PyTorch returns 0 for atan2(0, 0)
    assert torch.allclose(torch_result, torch.tensor([0.0])), \
        f"Expected torch.atan2(0, 0) to be 0, but got {torch_result.item()}"

    # 2. Test ONNX Exported Behavior
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=True) as tmp_file:
        onnx_path = tmp_file.name
        
        # Export model to ONNX
        torch.onnx.export(
            net, 
            (x, y), 
            onnx_path, 
            input_names=["x", "y"], 
            output_names=["output"],
            opset_version=14
        )

        # Run inference with ONNX Runtime
        sess = ort.InferenceSession(onnx_path)
        ort_result = sess.run(["output"], {"x": x.numpy(), "y": y.numpy()})[0]

        # Assert ONNX result matches PyTorch result (0) and is not NaN
        assert not np.isnan(ort_result).any(), \
            f"ONNX export produced NaN for atan2(0, 0), expected 0"
        
        assert np.allclose(torch_result.numpy(), ort_result), \
            f"Mismatch between PyTorch ({torch_result.item()}) and ONNX ({ort_result[0]}) results"

    print("test_atan2_zero_input_consistency passed.")

def test_angle_zero_input_consistency():
    """
    Test that torch.angle(0+0j) returns 0 and that the ONNX export
    produces the same result. torch.angle decomposes to atan2(imag, real).
    """
    
    class AngleNet(nn.Module):
        def forward(self, x):
            return torch.angle(x)

    # 1. Test Native PyTorch Behavior
    net = AngleNet()
    # Input is 0 + 0j
    x = torch.tensor([0.0 + 0.0j])
    
    torch_result = net(x)
    
    # Assert PyTorch returns 0 for angle(0+0j)
    assert torch.allclose(torch_result, torch.tensor([0.0])), \
        f"Expected torch.angle(0+0j) to be 0, but got {torch_result.item()}"

    # 2. Test ONNX Exported Behavior
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=True) as tmp_file:
        onnx_path = tmp_file.name
        
        # Export model to ONNX
        torch.onnx.export(
            net, 
            (x,), 
            onnx_path, 
            input_names=["x"], 
            output_names=["output"],
            opset_version=14
        )

        # Run inference with ONNX Runtime
        sess = ort.InferenceSession(onnx_path)
        ort_result = sess.run(["output"], {"x": x.numpy()})[0]

        # Assert ONNX result matches PyTorch result (0) and is not NaN
        assert not np.isnan(ort_result).any(), \
            f"ONNX export produced NaN for angle(0+0j), expected 0"
        
        assert np.allclose(torch_result.numpy(), ort_result), \
            f"Mismatch between PyTorch ({torch_result.item()}) and ONNX ({ort_result[0]}) results"

    print("test_angle_zero_input_consistency passed.")

if __name__ == "__main__":
    test_atan2_zero_input_consistency()
    test_angle_zero_input_consistency()