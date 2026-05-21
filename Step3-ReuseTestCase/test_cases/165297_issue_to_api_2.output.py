import unittest
import torch
import torch.nn as nn

class TestMaxPool2dChannelsLast(unittest.TestCase):
    """
    Test cases for Issue 165297: MaxPool2d with channels_last + bfloat16 on CUDA produces NaNs.
    The similar API (tf.feature_column.crossed_column) is semantically unrelated to this 
    PyTorch CUDA memory layout bug, so the test focuses on the original API under test.
    """

    def setUp(self):
        self.device = torch.device("cuda")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

    def test_maxpool_bfloat16_channels_last_nan(self):
        """
        Reproduce the NaN issue with bfloat16 and channels_last memory format.
        """
        N, C, H, W = 84, 64, 512, 960
        # Create input tensor with bfloat16
        x = torch.randn(N, C, H, W, dtype=torch.bfloat16, device=self.device)
        
        # Convert to channels_last (NHWC) memory format
        x = x.to(memory_format=torch.channels_last)

        pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1).to(self.device)
        y = pool(x)

        # Check for NaNs in the output
        self.assertFalse(torch.isnan(y).any(), 
                         "Output contains NaNs with bfloat16 and channels_last memory format")

    def test_maxpool_float32_channels_last_stability(self):
        """
        Check for illegal memory access or NaNs with float32 and channels_last.
        Note: Illegal memory access might cause a hard crash (segfault) which 
        unittest cannot catch, but we check for NaNs if it runs.
        """
        N, C, H, W = 84, 64, 512, 960
        # Create input tensor with float32
        x = torch.randn(N, C, H, W, dtype=torch.float32, device=self.device)
        
        # Convert to channels_last (NHWC) memory format
        x = x.to(memory_format=torch.channels_last)

        pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1).to(self.device)
        
        try:
            y = pool(x)
            self.assertFalse(torch.isnan(y).any(), 
                             "Output contains NaNs with float32 and channels_last memory format")
        except RuntimeError as e:
            # Catch potential CUDA errors related to memory access
            self.fail(f"RuntimeError (potential illegal memory access) occurred: {e}")

    def test_maxpool_contiguous_workaround(self):
        """
        Verify that the contiguous (NCHW) format works correctly as a workaround.
        """
        N, C, H, W = 84, 64, 512, 960
        x = torch.randn(N, C, H, W, dtype=torch.bfloat16, device=self.device)
        
        # Ensure contiguous memory format
        x = x.contiguous()

        pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1).to(self.device)
        y = pool(x)

        self.assertFalse(torch.isnan(y).any(), 
                         "Output contains NaNs with contiguous format (workaround failed)")

if __name__ == '__main__':
    unittest.main()