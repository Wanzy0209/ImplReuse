import unittest
import torch
import torch.nn.functional as F

class TestReflectPaddingLargeDim(unittest.TestCase):
    """
    Test case for Issue 165861: Reflect padding CUDA error when 
    one of the batch dimensions is larger than uint16 max value (2**16).
    """

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_reflect_pad_large_batch_dim(self):
        # Test case where the first dimension is exactly 2**16
        # This was causing a CUDA error in the bug report
        x = torch.rand(2**16, 2, device="cuda")
        
        # Test reflect mode (the failing mode)
        # We expect this to succeed without raising a CUDA error
        try:
            out = F.pad(x, (1, 1), mode="reflect")
            self.assertEqual(out.shape, torch.Size([2**16, 4]))
        except RuntimeError as e:
            self.fail(f"Reflect padding failed with dim=2**16: {e}")

        # Verify other modes still work (regression check)
        F.pad(x, (1, 1), mode="constant")
        F.pad(x, (1, 1), mode="replicate")
        # Circular padding requires padding < dimension size, so use (0, 0) for this case
        F.pad(x, (0, 0), mode="circular")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_reflect_pad_large_mid_dim(self):
        # Test case where a middle dimension is 2**16
        x = torch.rand(1, 2**16, 2, device="cuda")
        
        try:
            out = F.pad(x, (1, 1), mode="reflect")
            self.assertEqual(out.shape, torch.Size([1, 2**16, 4]))
        except RuntimeError as e:
            self.fail(f"Reflect padding failed with mid dim=2**16: {e}")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_reflect_pad_dim_below_limit(self):
        # Control case: dimension is 2**16 - 1, should work fine
        x = torch.rand(2**16 - 1, 200, device="cuda")
        out = F.pad(x, (1, 1), mode="reflect")
        self.assertEqual(out.shape, torch.Size([2**16 - 1, 202]))

if __name__ == '__main__':
    unittest.main()