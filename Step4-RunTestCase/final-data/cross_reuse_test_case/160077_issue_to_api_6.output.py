import torch
import unittest

# The similar API (tf.keras.initializers.LecunUniform) utilizes a class-based structure 
# inheriting from a parent class to configure and apply logic. 
# We leverage this pattern here by wrapping the torch.split logic in a torch.nn.Module 
# to test the Original API (torch.split) under the reported conditions.

class SplitModule(torch.nn.Module):
    """
    A module wrapper for torch.split, mimicking the class-based structure 
    of the similar API (tf.keras.initializers.LecunUniform).
    """
    def __init__(self, split_size, dim=0):
        # Mimicking the super().__init__ pattern seen in the similar API
        super(SplitModule, self).__init__()
        self.split_size = split_size
        self.dim = dim

    def forward(self, xs):
        # Original API Under Test: torch.split (called as a method on the tensor)
        return xs.split(self.split_size, dim=self.dim)

class TestTorchCompileSplit(unittest.TestCase):
    def test_split_compile_with_device_context(self):
        """
        Reproduces the bug: 'torch._tensor' has no attribute 'split' 
        while using torch.compile under torch.device context.
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping test.")

        # Fix: torch.device context manager is not available in older PyTorch versions.
        # Use torch.cuda.device context manager instead to maintain the context logic.
        with torch.cuda.device(torch.cuda.current_device()):
            xs = torch.randn(2, 2, device="cuda")
            model = SplitModule(split_size=1, dim=0)

            # 1. Verify eager execution works (as per bug report)
            eager_result = model(xs)
            self.assertEqual(len(eager_result), 2)

            # 2. Verify compiled execution works (this is where the bug occurred)
            # The bug was that torch.compile generated a call to torch._tensor.split
            # which does not exist as a module attribute.
            try:
                compiled_model = torch.compile(model)
                compiled_result = compiled_model(xs)
                
                # Validate results match
                self.assertEqual(len(compiled_result), 2)
                for e_tensor, c_tensor in zip(eager_result, compiled_result):
                    self.assertTrue(torch.allclose(e_tensor, c_tensor))
            except AttributeError as e:
                if "torch._tensor" in str(e) and "split" in str(e):
                    self.fail(f"Bug reproduced: {e}")
                else:
                    raise

if __name__ == '__main__':
    unittest.main()