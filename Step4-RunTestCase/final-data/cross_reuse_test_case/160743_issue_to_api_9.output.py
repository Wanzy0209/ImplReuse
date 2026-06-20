import torch
import unittest

class TestMPSAvgPool2d(unittest.TestCase):
    def test_avgpool2d_mps_divisor_override_ceil_mode(self):
        """
        Test case for Issue 160743: MPS AvgPool2d produces incorrect output.
        
        This test verifies that torch.nn.AvgPool2d with specific parameters
        (kernel_size, stride, ceil_mode, divisor_override) produces consistent
        results between CPU and MPS backends.
        
        The logic mirrors the pattern of checking a feature flag (similar to
        tf.compat.v1.resource_variables_enabled) by verifying MPS availability
        before executing the backend-specific comparison.
        """
        # Check if MPS is available, similar to checking if a resource variable feature is enabled
        if not torch.backends.mps.is_available():
            self.skipTest("MPS backend is not available on this system.")

        torch.manual_seed(0)

        # Initialize model with parameters from the bug report
        model = torch.nn.AvgPool2d(
            kernel_size=[1, 6], 
            stride=[4, 9], 
            ceil_mode=True, 
            divisor_override=3
        )
        
        # Generate input tensor
        x = torch.randn(4, 6, 7)

        # Compute output on CPU
        out_cpu = model(x)

        # Compute output on MPS
        # Note: We must move the model to MPS as well
        model_mps = model.to("mps")
        x_mps = x.to("mps")
        out_mps = model_mps(x_mps)

        # Assert that the outputs are close within a reasonable tolerance
        # The bug report shows discrepancies where MPS output becomes 0.0
        self.assertTrue(
            torch.allclose(out_cpu, out_mps.cpu(), atol=1e-2, rtol=1e-2),
            f"Output does not match between CPU and MPS!\nCPU:\n{out_cpu}\nMPS:\n{out_mps.cpu()}"
        )

if __name__ == "__main__":
    unittest.main()