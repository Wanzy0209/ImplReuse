import torch
from torch import nn
import unittest

class TestJitScriptCompilerDisable(unittest.TestCase):
    """
    Regression test for Issue #160059.
    Verifies that torch.jit.script works correctly when a module's forward method
    is decorated with @torch.compiler.disable.
    """

    @unittest.skipIf(not hasattr(torch, 'compiler'), "torch.compiler not available in this PyTorch version")
    def test_jit_script_with_compiler_disable(self):
        # Helper function defined outside the module to test symbol resolution
        def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
            x = x.clamp(min=0, max=1)
            x1 = x.clamp(min=eps)
            x2 = (1 - x).clamp(min=eps)
            return torch.log(x1 / x2)

        class Model(nn.Module):
            @torch.compiler.disable(recursive=False)
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Call to external function to test scope handling
                return inverse_sigmoid(x)

        model = Model()
        
        # This call should not raise a RuntimeError about 'undefined value'
        # It was failing in PyTorch 2.8.0 due to a regression
        try:
            scripted_model = torch.jit.script(model)
        except RuntimeError as e:
            self.fail(f"torch.jit.script failed with @torch.compiler.disable: {e}")

        # Verify the scripted model executes correctly
        input_tensor = torch.tensor([0.2, 0.5, 0.8])
        output = scripted_model(input_tensor)
        
        # Basic assertions to ensure functionality
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, input_tensor.shape)
        self.assertTrue(torch.isfinite(output).all())

if __name__ == "__main__":
    unittest.main()