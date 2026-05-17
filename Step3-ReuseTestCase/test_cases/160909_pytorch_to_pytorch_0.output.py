import torch
import torch.nn as nn
import torch.utils.privateuse1

# Define a simple model that triggers the specific operation mentioned in the bug report
class SimpleModel(nn.Module):
    def forward(self, x):
        # The error mentions repeat_interleave calling flatten
        return torch.repeat_interleave(x, repeats=2, dim=1)

# Define a dummy backend for compilation
def my_backend(gm, example_inputs):
    # The bug report states the error happens before the backend logic is fully executed,
    # but we return the forward pass to satisfy the API signature.
    return gm.forward

def test_compile_privateuse1_meta_tensor_bug():
    device_type = torch.utils.privateuse1.device_type
    
    model = SimpleModel()
    # Move model to PrivateUse1
    model = model.to(device_type)
    
    # Create dummy data on PrivateUse1
    # Shape matches the error log: (1, 8, 3, 128)
    data = torch.randn(1, 8, 3, 128).to(device_type)
    
    # Attempt to compile
    # The bug is that torch.compile passes meta tensors to PrivateUse1 ops during tracing
    try:
        compiled_model = torch.compile(model, backend=my_backend)
        # This call triggers the tracing where the bug occurs
        result = compiled_model(data)
        print("Test passed (or bug is fixed)")
    except RuntimeError as e:
        error_msg = str(e)
        # Check for the specific error message from the bug report
        if "storage is not on the custom PrivateUse1 device" in error_msg and "meta" in error_msg:
            print(f"Bug reproduced: {error_msg}")
        else:
            # Re-raise if it's a different error (e.g. missing kernels or other issues)
            raise

if __name__ == "__main__":
    test_compile_privateuse1_meta_tensor_bug()