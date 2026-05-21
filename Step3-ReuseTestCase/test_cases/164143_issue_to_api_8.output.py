import torch
import torch._dynamo as dynamo
import unittest

# Leveraging the similar API pattern (tf.keras.initializers.HeNormal) 
# by porting the class structure to PyTorch to be used within the test.
class VarianceScaling:
    def __init__(self, scale=1.0, mode='fan_in', distribution='truncated_normal', seed=None):
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed

class HeNormal(VarianceScaling):
    def __init__(self, seed=None):
        super(HeNormal, self).__init__(
            scale=2., mode='fan_in', distribution='truncated_normal', seed=seed)

    def __call__(self, shape, dtype=None):
        if self.seed is not None:
            torch.manual_seed(self.seed)
        # Simplified implementation for testing purposes
        return torch.randn(shape) * (self.scale ** 0.5)

class TestDebugModeCompile(unittest.TestCase):
    def test_compile_with_debug_mode_raises_error(self):
        """
        Test that torch.compile raises an error when DebugMode is active,
        rather than silently skipping compilation.
        """
        # Define a function that utilizes the reused HeNormal pattern
        def model_fn(x):
            initializer = HeNormal(seed=42)
            # Create a weight using the initializer
            w = initializer((x.shape[1], x.shape[1]))
            return torch.matmul(x, w)

        input_tensor = torch.randn(5, 5)

        # The bug report indicates that when a non-infra torch dispatch mode (like DebugMode)
        # is present, torch.compile silently skips compilation.
        # The expected behavior (fix) is to raise an error.
        
        # We use torch._C._DebugModeGuard to simulate the environment described in the bug.
        with torch._C._DebugModeGuard():
            with self.assertRaises(RuntimeError) as context:
                compiled_fn = torch.compile(model_fn)
                compiled_fn(input_tensor)
            
            # Verify the error message relates to the dispatch mode issue
            self.assertIn("non-infra torch dispatch mode", str(context.exception))

if __name__ == "__main__":
    unittest.main()