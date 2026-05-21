import torch
import torch.nn as nn
import onnxruntime as ort
import tempfile
import os

# This helper function mirrors the structure of the similar API 
# (tensorflow.compiler.tests.xla_call_module_test.serialize),
# abstracting the serialization/export and execution process.
def serialize_and_run(model, inputs, input_names, output_names):
    """
    Serializes the model to ONNX and runs inference, mimicking the 
    pattern of the similar API's serialize function.
    """
    # Create a temporary file for the ONNX export
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        onnx_path = tmp.name

    try:
        # Export the model
        torch.onnx.export(
            model, 
            inputs, 
            onnx_path, 
            input_names=input_names, 
            output_names=output_names
        )
        
        # Run inference using ONNX Runtime
        sess = ort.InferenceSession(onnx_path)
        ort_inputs = {name: inp.numpy() for name, inp in zip(input_names, inputs)}
        result = sess.run(output_names, ort_inputs)[0]
        
        return result
    finally:
        # Clean up the temporary file
        if os.path.exists(onnx_path):
            os.remove(onnx_path)

class Net(nn.Module):
    def forward(self, x, y):
        return torch.atan2(x, y)

def test_atan2_zero_input_onnx_consistency():
    """
    Test case to verify that torch.atan2(0, 0) produces the same result
    in PyTorch and ONNX export.
    Bug: ONNX export produces NaN while PyTorch produces 0.
    """
    net = Net()
    
    # Edge case inputs that trigger the bug
    x = torch.tensor([0.0])
    y = torch.tensor([0.0])
    
    # Native PyTorch result
    torch_result = net(x, y)
    
    # Result from ONNX export (using the helper pattern)
    ort_result = serialize_and_run(net, (x, y), ["x", "y"], ["output"])
    
    print(f"PyTorch result: {torch_result}")
    print(f"ONNX Runtime result: {ort_result}")
    
    # Assert that the results are consistent.
    # torch.allclose returns False if comparing NaN with a number, which captures the bug.
    assert torch.allclose(torch_result, torch.tensor(ort_result), equal_nan=True), \
        f"ONNX export mismatch: PyTorch={torch_result.item()}, ONNX={ort_result[0]}"

if __name__ == "__main__":
    test_atan2_zero_input_onnx_consistency()