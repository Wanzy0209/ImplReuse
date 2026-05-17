import torch
import torch.onnx
import tempfile
import os

def test_onnx_export_with_none_output():
    """
    Test case to verify that torch.onnx.export handles models returning tuples 
    containing None values without crashing.
    
    Relates to Issue: [ONNX] Exporter crashes when fx node output includes None
    """
    
    class ModelWithNoneOutput(torch.nn.Module):
        def __init__(self, return_dict=True):
            super().__init__()
            self.return_dict = return_dict

        def forward(self, x):
            # Simulate the logic from the bug report
            output = x + 1
            image_tokens_masks = None  # This variable can be None
            
            if not self.return_dict:
                # The bug occurs here when exporting a graph with this return statement
                return (output, image_tokens_masks)
            
            return {"output": output}

    # Create a model instance that returns the tuple with None
    model = ModelWithNoneOutput(return_dict=False)
    dummy_input = torch.randn(1, 3, 32, 32)

    # Create a temporary file for the export
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=True) as tmp_file:
        onnx_path = tmp_file.name

    try:
        # Attempt to export the model
        # In the bug scenario, this would crash because the exporter 
        # did not consider None as a valid node output.
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output", "mask"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        
        # Verify the file was created
        assert os.path.exists(onnx_path), "ONNX file was not created"
        print("Test Passed: Exporter handled None in output tuple successfully.")

    except Exception as e:
        print(f"Test Failed: {e}")
        raise

if __name__ == "__main__":
    test_onnx_export_with_none_output()