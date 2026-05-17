import torch
import unittest

class TestMPSAvgPool2d(unittest.TestCase):
    """
    Test case for Issue 160743: MPS AvgPool2d produces incorrect output.
    
    This test reproduces the bug where torch.nn.AvgPool2d with specific parameters
    (kernel_size=[1, 6], stride=[4, 9], ceil_mode=True, divisor_override=3)
    produces mismatched results between CPU and MPS backends.
    
    The test leverages the semantic pattern of the similar API (tf.sysconfig.get_build_info)
    by checking the build/environment availability of the MPS backend before execution.
    """

    def test_avgpool2d_mps_cpu_consistency(self):
        # Leverage similar API pattern: Check build/environment info
        # In PyTorch, we check if the MPS backend is built and available.
        if not torch.backends.mps.is_available():
            self.skipTest("MPS backend is not available or not built on this system.")

        torch.manual_seed(0)

        # Model definition from the bug report
        model = torch.nn.AvgPool2d(
            kernel_size=[1, 6], 
            stride=[4, 9], 
            ceil_mode=True, 
            divisor_override=3
        )
        
        # Input tensor (3D: C, H, W)
        x = torch.randn(4, 6, 7)
        
        # Compute output on CPU
        out_cpu = model(x)
        
        # Compute output on MPS
        x_mps = x.to("mps")
        out_mps = model(x_mps)
        
        # Convert MPS output back to CPU for comparison
        out_mps_cpu = out_mps.cpu()

        # Assert that the outputs are close within a reasonable tolerance
        # The bug report shows specific values becoming 0.0 on MPS, which should fail this check.
        try:
            self.assertTrue(
                torch.allclose(out_cpu, out_mps_cpu, atol=1e-2, rtol=1e-2),
                "MPS output does not match CPU output."
            )
        except AssertionError as e:
            print("Output does not match!")
            print("CPU Output:\n", out_cpu)
            print("MPS Output:\n", out_mps_cpu)
            raise e

if __name__ == '__main__':
    unittest.main()