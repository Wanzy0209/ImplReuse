import torch
import unittest

class TestTorchAnyCompile(unittest.TestCase):
    def test_torch_any_compiled(self):
        """
        Test that torch.any works correctly within a torch.compile context.
        Adapted from the context of test_dtensor_compile_redistribute
        which involves compiling operations on tensors.
        """
        def fn(x):
            # torch.any is a reduction operation, similar to operations
            # that might be involved in redistribution logic.
            return torch.any(x > 0)

        # Input tensor
        x = torch.randn(10)

        # Compile the function
        compiled_fn = torch.compile(fn)

        # Execute and verify
        res = compiled_fn(x)
        expected = fn(x)

        self.assertEqual(res, expected)

    def test_torch_any_all_false(self):
        def fn(x):
            return torch.any(x)

        x = torch.zeros(10, dtype=torch.bool)
        compiled_fn = torch.compile(fn)
        
        res = compiled_fn(x)
        expected = fn(x)
        
        self.assertEqual(res, expected)

if __name__ == "__main__":
    unittest.main()