import unittest
import torch
import torch.nn.functional as F

class TestReflectPaddingLargeBatch(unittest.TestCase):
    """
    Test case for Issue 165861: Reflect padding CUDA error when batch dimension > 2**16.
    
    This test mirrors the pattern of checking environment capabilities (similar to 
    tf.test.is_built_with_xla) before executing hardware-specific tests.
    """

    @classmethod
    def setUpClass(cls):
        # Mirroring the pattern of checking build support (is_built_with_xla)
        # by checking CUDA availability before running tests.
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available, skipping CUDA-specific test")

    def test_reflect_pad_at_limit(self):
        """Test reflect padding with batch dimension exactly at uint16 max (2**16)."""
        # Dimension size exactly 2**16
        # Changed last dimension from 2 to 3 to avoid circular padding error with small dimensions
        x = torch.rand(2**16, 3, device="cuda")
        
        # Verify other modes work (baseline)
        F.pad(x, (1, 1), mode="constant")
        F.pad(x, (1, 1), mode="circular")
        F.pad(x, (1, 1), mode="replicate")
        
        # Test the specific failing mode: reflect
        # This should not raise a CUDA error
        try:
            result = F.pad(x, (1, 1), mode="reflect")
            self.assertEqual(result.shape, (2**16, 5))
        except RuntimeError as e:
            self.fail(f"Reflect padding failed with batch dim 2**16: {e}")

    def test_reflect_pad_above_limit(self):
        """Test reflect padding with batch dimension slightly above uint16 max."""
        # Dimension size 2**16 + 1
        # Changed last dimension from 2 to 3 to avoid circular padding error with small dimensions
        x = torch.rand(2**16 + 1, 3, device="cuda")
        
        try:
            result = F.pad(x, (1, 1), mode="reflect")
            self.assertEqual(result.shape, (2**16 + 1, 5))
        except RuntimeError as e:
            self.fail(f"Reflect padding failed with batch dim 2**16 + 1: {e}")

    def test_reflect_pad_multi_dim_limit(self):
        """Test reflect padding when a non-first dimension hits the limit."""
        # Shape (1, 2**16, 3)
        # Changed last dimension from 2 to 3 to avoid circular padding error with small dimensions
        x = torch.rand(1, 2**16, 3, device="cuda")
        
        try:
            result = F.pad(x, (1, 1), mode="reflect")
            self.assertEqual(result.shape, (1, 2**16, 5))
        except RuntimeError as e:
            self.fail(f"Reflect padding failed with shape (1, 2**16, 3): {e}")

    def test_reflect_pad_below_limit(self):
        """Test reflect padding with batch dimension just below the limit (sanity check)."""
        # Dimension size 2**16 - 1
        x = torch.rand(2**16 - 1, 200, device="cuda")
        
        try:
            result = F.pad(x, (1, 1), mode="reflect")
            self.assertEqual(result.shape, (2**16 - 1, 202))
        except RuntimeError as e:
            self.fail(f"Reflect padding failed with batch dim 2**16 - 1: {e}")

if __name__ == '__main__':
    unittest.main()