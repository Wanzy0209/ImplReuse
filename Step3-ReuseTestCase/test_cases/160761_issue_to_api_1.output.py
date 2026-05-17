import torch
import torch.nn as nn

class VmapWrapperModel(nn.Module):
    """
    A minimal model that mimics the usage of torch.vmap found in 
    transformers masking_utils.py, which triggers the ONNX export bug.
    """
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        # Define a function to be vmapped
        def _forward_single(sample):
            return self.linear(sample)
        
        # Use torch.vmap to apply the function across the batch dimension
        # This is the pattern causing issues with torch.onnx.export
        return torch.vmap(_forward_single)(x)

def test_gemma3_vmap_onnx_export():
    """
    Reproduces the issue where torch.onnx.export fails due to torch.vmap.
    Also leverages torch.backends.cusparselt.version to check backend state,
    following the pattern of the similar API.
    """
    
    # 1. Leverage the similar API: Check backend version
    # This reflects the pattern of checking initialization/availability
    # seen in torch.backends.cusparselt.version implementation.
    try:
        backend_version = torch.backends.cusparselt.version()
        print(f"cuSPARSELt version: {backend_version}")
    except (AttributeError, RuntimeError):
        print("cuSPARSELt backend not available or not initialized.")
        backend_version = None

    # 2. Setup the model and inputs
    model = VmapWrapperModel()
    model.eval()
    
    # Input tensor: Batch size 2, dimension 10
    dummy_input = torch.randn(2, 10)

    # 3. Attempt ONNX Export (The failing operation from the issue)
    # The issue reports failure both with and without dynamo.
    # We test with dynamo=True as it was explicitly mentioned.
    try:
        torch.onnx.export(
            model,
            dummy_input,
            "vmap_repro.onnx",
            input_names=["input_ids"],
            output_names=["logits"],
            opset_version=17,
            dynamo=True
        )
        print("ONNX Export succeeded.")
    except Exception as e:
        # The bug report indicates this export fails.
        # We catch the exception to demonstrate the reproduction logic.
        print(f"ONNX Export failed with error: {e}")

if __name__ == "__main__":
    test_gemma3_vmap_onnx_export()