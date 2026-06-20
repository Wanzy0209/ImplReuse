import torch
import torch.nn as nn
import pytest

# Fix: Check if torch.compiler exists to avoid AttributeError in older PyTorch versions
# or environments where this feature is not available.
if not hasattr(torch, 'compiler'):
    pytest.skip("torch.compiler is not available in this version of PyTorch", allow_module_level=True)

def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Helper function used in the model."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

class Model(nn.Module):
    """Model with a forward method decorated by torch.compiler.disable."""
    @torch.compiler.disable(recursive=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = inverse_sigmoid(x)
        return x

def test_jit_script_with_compiler_disable():
    """
    Regression test for Issue 160059.
    
    Verifies that torch.jit.script does not fail with an 'undefined value' error
    when the module's forward method is decorated with @torch.compiler.disable.
    """
    model = Model()
    
    # This call was failing in PyTorch 2.8.0 with an 'undefined value' error.
    # We expect it to succeed and return a ScriptModule.
    scripted_model = torch.jit.script(model)
    
    # Verify the scripted model is functional
    input_tensor = torch.tensor([0.5, 0.1, 0.9])
    output = scripted_model(input_tensor)
    
    # Basic assertion to ensure execution completed
    assert output is not None
    assert output.shape == input_tensor.shape

if __name__ == "__main__":
    test_jit_script_with_compiler_disable()
    print("Test passed.")