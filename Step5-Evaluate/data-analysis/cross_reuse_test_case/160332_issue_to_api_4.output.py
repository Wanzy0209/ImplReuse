import torch
import unittest
from torch import nn

class TestConvTranspose3DMPSAutocast(unittest.TestCase):
    def test_convtranspose3d_mps_autocast_fp32(self):
        """
        Test that ConvTranspose3D works with autocast on MPS by using FP32,
        as FP16/BF16 are not supported.
        
        This test leverages the pattern of checking backend capabilities
        similar to torch.backends.cudnn.version.
        """
        # Check if MPS is available, analogous to checking CUDA/cuDNN availability
        if not torch.backends.mps.is_available():
            self.skipTest("MPS backend is not available")

        device = torch.device('mps')

        # Reproduce the original bug logic
        with torch.amp.autocast(device_type=device.type):
            m = nn.ConvTranspose3d(16, 33, 3, stride=2)
            m.to(device)
            x = torch.randn(20, 16, 10, 50, 100).to(device)
            
            # This should not raise RuntimeError:
            # "ConvTranspose 3D with BF16 or FP16 types is not supported on MPS"
            output = m(x)

            # Verify the output shape is correct
            self.assertEqual(output.shape, (20, 33, 21, 101, 201))
            
            # Verify the output dtype is FP32. Since autocast defaults to FP16/BF16
            # but the op does not support it, it should fallback to FP32.
            self.assertEqual(output.dtype, torch.float32)

if __name__ == '__main__':
    unittest.main()