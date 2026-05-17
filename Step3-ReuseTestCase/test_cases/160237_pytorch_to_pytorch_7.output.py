import torch
import torch.nn.functional as F
import unittest

class TestConv3dMPS(unittest.TestCase):
    def test_conv3d_on_mps_device(self):
        """
        Test case for torch.nn.functional.conv3d on MPS device.
        Based on the bug report for grid_sampler_3d not being implemented on MPS,
        this test verifies the similar API (conv3d) on the same device.
        """
        if not torch.backends.mps.is_available():
            self.skipTest("MPS device is not available")

        # Setup input and weight tensors mimicking 3D data processing
        # Shape: (Batch, Channels, Depth, Height, Width)
        batch_size = 1
        in_channels = 3
        out_channels = 16
        depth, height, width = 10, 32, 32
        kernel_size = 3

        input_tensor = torch.randn(batch_size, in_channels, depth, height, width)
        weight_tensor = torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size)

        # Move tensors to MPS device
        input_mps = input_tensor.to("mps")
        weight_mps = weight_tensor.to("mps")

        # Execute the similar API: conv3d
        # This replaces the grid_sample call from the original bug context
        try:
            output = F.conv3d(input_mps, weight_mps)
            
            # Assertions to verify correctness
            self.assertIsNotNone(output)
            # Expected output size calculation: (D - K + 1)
            expected_depth = depth - kernel_size + 1
            expected_height = height - kernel_size + 1
            expected_width = width - kernel_size + 1
            expected_shape = (batch_size, out_channels, expected_depth, expected_height, expected_width)
            
            self.assertEqual(output.shape, torch.Size(expected_shape))
            self.assertEqual(output.device.type, "mps")

        except NotImplementedError as e:
            self.fail(f"conv3d is not implemented for MPS device: {e}")

if __name__ == "__main__":
    unittest.main()