import torch
from torch import nn

def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Helper function used in the model forward pass.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

class Model(nn.Module):
    """
    Model class with a forward method decorated by @torch.compiler.disable.
    This pattern caused a regression in PyTorch 2.8.0 when used with torch.jit.script.
    """
    @torch.compiler.disable(recursive=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = inverse_sigmoid(x)
        return x

def test_torch_jit_script_with_compiler_disable():
    """
    Regression test for Issue #160059.
    
    Verifies that torch.jit.script successfully compiles a module
    when the forward method is decorated with @torch.compiler.disable.
    
    The test ensures that the interaction between the compiler decorator
    and the scripting environment (similar to context handling in 
    tf.keras.backend.set_value) does not raise an 'undefined value' error.
    """
    model = Model()
    
    # This call raised 'undefined value' error in PyTorch 2.8.0
    # It should complete without exceptions.
    try:
        scripted_model = torch.jit.script(model)
    except Exception as e:
        pytest.fail(f"torch.jit.script failed with error: {e}")

    # Verify the scripted model is functional
    dummy_input = torch.tensor([0.5, 0.5])
    output = scripted_model(dummy_input)
    
    assert output is not None
    assert output.shape == dummy_input.shape

if __name__ == "__main__":
    # Basic execution check if pytest is not used
    test_torch_jit_script_with_compiler_disable()
    print("Test passed.")