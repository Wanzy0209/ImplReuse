import torch
import unittest
import sys

class TestGridSampler3DNaNHandling(unittest.TestCase):
    def test_grid_sampler_3d_nan_mps_vs_cpu(self):
        """
        Test that grid_sampler_3d correctly propagates NaN values on the MPS backend,
        matching the CPU behavior. 
        Ref: Issue #163851
        """
        # Skip if MPS is not available
        if not torch.backends.mps.is_available():
            self.skipTest("MPS backend not available")

        # Setup input and grid with NaN
        # Input shape: (N, C, D, H, W) = (1, 1, 3, 3, 3)
        input_tensor = torch.ones(1, 1, 3, 3, 3)
        
        # Grid shape: (N, C, D, H, W) = (1, 1, 1, 2, 3)
        # Contains a NaN value at the first position
        grid_nan = torch.tensor([[[[[torch.nan, 1., 1.], [1., 1., 1.]]]]])

        # Compute on CPU
        out_cpu = torch.grid_sampler_3d(input_tensor, grid_nan, 0, 0, True)

        # Compute on MPS
        input_mps = input_tensor.to("mps")
        grid_mps = grid_nan.to("mps")
        out_mps = torch.grid_sampler_3d(input_mps, grid_mps, 0, 0, True)

        # Check that NaNs are present in the CPU output
        self.assertTrue(torch.isnan(out_cpu).any(), "CPU output should contain NaN")

        # Check that NaNs are present in the MPS output (The bug fix)
        # The original bug was that MPS returned 1.0 instead of NaN
        self.assertTrue(torch.isnan(out_mps).any(), "MPS output should contain NaN")

        # Verify that the pattern of NaNs matches between CPU and MPS
        cpu_is_nan = torch.isnan(out_cpu)
        mps_is_nan = torch.isnan(out_mps.cpu())
        
        self.assertTrue(torch.equal(cpu_is_nan, mps_is_nan), 
                        "MPS and CPU should have NaNs in the same locations")

        # Verify that non-NaN values are close
        # Create a mask where values are not NaN on either device
        mask = ~cpu_is_nan
        if mask.any():
            cpu_vals = out_cpu[mask]
            mps_vals = out_mps.cpu()[mask]
            torch.testing.assert_close(cpu_vals, mps_vals)

if __name__ == '__main__':
    unittest.main()