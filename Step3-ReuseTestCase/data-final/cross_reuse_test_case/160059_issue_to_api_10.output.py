import torch
import torch.nn as nn
import unittest

def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Helper function used within the model."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

class TestJitScriptCompilerDisable(unittest.TestCase):
    """
    Regression test for Issue #160059.
    
    Verifies that torch.jit.script does not fail with an 'undefined value' error
    when scripting a module decorated with @torch.compiler.disable.
    """

    def test_jit_script_with_compiler_disable(self):
        class Model(nn.Module):
            @torch.compiler.disable(recursive=False)
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return inverse_sigmoid(x)

        model = Model()
        
        # In PyTorch 2.8.0, this raised an "undefined value" error.
        # We assert that it completes successfully.
        try:
            scripted_model = torch.jit.script(model)
        except Exception as e:
            self.fail(f"torch.jit.script failed on module with @torch.compiler.disable: {e}")

        # Verify the scripted model is functional
        input_tensor = torch.tensor([0.5, 0.1, 0.9])
        output = scripted_model(input_tensor)
        
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, input_tensor.shape)

if __name__ == "__main__":
    unittest.main()