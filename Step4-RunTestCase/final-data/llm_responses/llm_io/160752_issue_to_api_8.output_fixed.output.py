import torch
import torch.nn.functional as F
import unittest

class TestJacfwdfailsWithOneHotAndCompile(unittest.TestCase):
    """
    Test case for Issue ID: 160752
    Verifies that torch.func.jacfwd works with one_hot and torch.compile(dynamic=True).
    """

    def setUp(self):
        # Skip tests if running on PyTorch < 2.0 where these features don't exist
        if not hasattr(torch, 'compile') or not hasattr(torch, 'func'):
            self.skipTest("PyTorch 2.0+ is required for torch.compile and torch.func")

        self.MAX = 3
        self.BATCH = 37
        # Generate deterministic inputs for reproducibility
        torch.manual_seed(42)
        self.idxs = torch.randint(self.MAX, (self.BATCH,), dtype=torch.int64)
        self.x = torch.rand((self.BATCH, self.MAX), dtype=torch.float64)

    def func(self, x, idxs):
        """Base function involving one_hot encoding."""
        return x.square() * F.one_hot(idxs, self.MAX)

    def jacfunc(self, x, idxs):
        """Jacobian forward pass of the base function."""
        return torch.func.jacfwd(self.func, argnums=(0,))(x, idxs)

    def test_eager_execution(self):
        """Test that the function works in eager mode (baseline)."""
        out = self.jacfunc(self.x, self.idxs)
        self.assertIsNotNone(out)
        # Check shape consistency
        self.assertEqual(out.shape, (self.BATCH, self.MAX, self.BATCH, self.MAX))

    def test_compiled_execution_dynamic(self):
        """
        Test that the function works with torch.compile(dynamic=True).
        This corresponds to the failing case in the bug report.
        """
        compiled_jacfunc = torch.compile(self.jacfunc, dynamic=True)
        
        try:
            out = compiled_jacfunc(self.x, self.idxs)
            self.assertIsNotNone(out)
            self.assertEqual(out.shape, (self.BATCH, self.MAX, self.BATCH, self.MAX))
        except Exception as e:
            self.fail(f"torch.compile(dynamic=True) failed with error: {e}")

    def test_compiled_vs_eager_consistency(self):
        """
        Test that compiled output matches eager execution output.
        This ensures the compilation does not alter semantics.
        """
        # Get eager result
        out_eager = self.jacfunc(self.x, self.idxs)
        
        # Get compiled result
        compiled_jacfunc = torch.compile(self.jacfunc, dynamic=True)
        out_compiled = compiled_jacfunc(self.x, self.idxs)
        
        # Assert values are close
        self.assertTrue(torch.allclose(out_eager, out_compiled, atol=1e-5))

if __name__ == "__main__":
    unittest.main()