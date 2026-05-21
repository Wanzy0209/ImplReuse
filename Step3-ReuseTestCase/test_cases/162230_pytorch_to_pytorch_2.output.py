import torch
import unittest

class TestScalarSubtract(unittest.TestCase):
    """
    Test case for torch.sub adapted from the context of test_scalar_multiply.
    This verifies scalar subtraction behavior under torch.compile (graph mode).
    """
    
    def test_scalar_subtract(self):
        # Test Tensor - Scalar
        def fn_tensor_scalar(x):
            return torch.sub(x, 2.5)

        x = torch.randn(5, 5)
        expected = fn_tensor_scalar(x)
        
        # Compile to verify graph execution (similar to FxGraphRunnableTest)
        compiled_fn = torch.compile(fn_tensor_scalar)
        actual = compiled_fn(x)
        
        self.assertTrue(torch.allclose(actual, expected))

        # Test Scalar - Tensor
        def fn_scalar_tensor(x):
            return torch.sub(2.5, x)

        expected = fn_scalar_tensor(x)
        compiled_fn = torch.compile(fn_scalar_tensor)
        actual = compiled_fn(x)
        
        self.assertTrue(torch.allclose(actual, expected))

if __name__ == "__main__":
    unittest.main()