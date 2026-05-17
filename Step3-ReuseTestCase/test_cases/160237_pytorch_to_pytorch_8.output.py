import torch
import torch.nn as nn
import unittest

class TestL1LossOnMPS(unittest.TestCase):
    def test_l1loss_mps_device(self):
        """
        Test case for torch.nn.L1Loss on the MPS device.
        This test is derived from a bug report where grid_sampler_3d failed on MPS.
        We verify that the similar API, L1Loss, functions correctly on the MPS backend.
        """
        # Check if MPS is available (Mac Metal)
        if not torch.backends.mps.is_available():
            self.skipTest("MPS device is not available. Skipping MPS-specific test.")

        device = torch.device("mps")

        # Create random input and target tensors
        # Simulating data flow that might occur in a ComfyUI node pipeline
        batch_size = 2
        channels = 3
        height = 64
        width = 64
        
        input_tensor = torch.randn(batch_size, channels, height, width, device=device)
        target_tensor = torch.randn(batch_size, channels, height, width, device=device)

        # Initialize the L1Loss (similar API)
        criterion = nn.L1Loss()

        # Execute the operation
        try:
            loss = criterion(input_tensor, target_tensor)
        except NotImplementedError as e:
            self.fail(f"L1Loss raised NotImplementedError on MPS device: {e}")

        # Verify the output
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Loss is a scalar by default
        self.assertEqual(loss.device.type, 'mps')
        self.assertGreaterEqual(loss.item(), 0.0)

if __name__ == '__main__':
    unittest.main()