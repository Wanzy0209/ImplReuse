import torch
import torch.nn.functional as F
import unittest

class TestInductorTanhRegression(unittest.TestCase):
    def test_compile_inductor_with_tanh(self):
        """
        Test case for Issue 160084: Regression on compile with backend inductor.
        This test adapts the original reproduction logic to leverage the 
        torch.nn.functional.tanh API to check if the regression affects 
        operations involving this specific function.
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Fix: Check if torch.compile exists (requires PyTorch 2.0+)
        if not hasattr(torch, 'compile'):
            self.skipTest("torch.compile is not available (requires PyTorch 2.0+)")

        class TanhRegressionModel(torch.nn.Module):
            def __init__(self, a=0, b=0):
                super().__init__()
                self.a = torch.nn.Parameter(torch.tensor(a).float())
                self.b = torch.nn.Parameter(torch.tensor(b).float())

            def forward(self, x=None):
                # Leverage the similar API: torch.nn.functional.tanh
                # Preserving the original logic structure (parameters + input)
                return F.tanh(x * self.a + self.b)
        
        model = TanhRegressionModel().cuda()
        
        # The regression occurs specifically with the 'inductor' backend
        model.forward = torch.compile(model.forward, backend="inductor")
        
        inputs = torch.randn(4, 10).cuda()
        
        # This call should not raise RuntimeError: opt_ready_stream && opt_parent_stream
        # If the bug exists, this line will trigger the assertion failure in the backend.
        output = model(inputs)
        
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (4, 10))

if __name__ == "__main__":
    unittest.main()