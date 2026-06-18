import torch
import torch.nn as nn
import unittest

class TestConvTranspose3DAutocastMPS(unittest.TestCase):
    """
    Test case to verify that ConvTranspose3D uses FP32 when autocast is enabled on MPS.
    This addresses the issue where autocast incorrectly attempts to use FP16/BF16,
    which is not supported for this operation on MPS.
    
    This test leverages the pattern of enabling a specific behavior (similar to 
    tf.compat.v1.enable_v2_behavior) and verifying the execution of operations 
    within that context.
    """

    def test_convtranspose3d_autocast_mps_fp32_fallback(self):
        # Skip if MPS is not available
        if not torch.backends.mps.is_available():
            self.skipTest("MPS backend is not available")

        device = torch.device('mps')

        # Enable autocast for the MPS device
        with torch.amp.autocast(device_type=device.type):
            # Verify that the autocast behavior is active, mirroring the 
            # explicit enable checks in the similar API (tf.compat.v1.enable_v2_behavior)
            self.assertTrue(torch.is_autocast_enabled(), "Autocast should be enabled within the context")

            # Reproduce the original bug logic
            m = nn.ConvTranspose3d(16, 33, 3, stride=2)
            m.to(device)
            x = torch.randn(20, 16, 10, 50, 100).to(device)
            
            # The operation should run without RuntimeError
            # The fix ensures that FP32 is used instead of the unsupported FP16/BF16
            u = m(x)
            
            # Assert that the output is indeed FP32
            self.assertEqual(u.dtype, torch.float32, 
                             "ConvTranspose3D on MPS with autocast should fallback to FP32")

if __name__ == '__main__':
    unittest.main()