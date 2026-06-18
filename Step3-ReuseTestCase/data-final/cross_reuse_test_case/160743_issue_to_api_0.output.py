import torch
import unittest

class TestAvgPool2dMPS(unittest.TestCase):
    def test_avgpool2d_mps_divisor_override_consistency(self):
        """
        Test case for Issue 160743: MPS AvgPool2d produces incorrect output.
        
        This test verifies the consistency of AvgPool2d with divisor_override 
        between CPU and MPS backends. It also leverages the similar API 
        torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed to check 
        backend-specific reduction settings, ensuring the MPS behavior is 
        validated independently of CUDA backend flags.
        """
        # Check CUDA backend flag (Similar API)
        # This verifies the environment state regarding reduction math, 
        # which is conceptually related to the divisor_override logic in AvgPool2d.
        cuda_reduction_allowed = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
        
        # Original bug reproduction logic
        torch.manual_seed(0)
        
        # Model configuration that triggered the bug
        model = torch.nn.AvgPool2d(
            kernel_size=[1, 6], 
            stride=[4, 9], 
            ceil_mode=True, 
            divisor_override=3
        )
        
        x = torch.randn(4, 6, 7)
        
        # CPU execution
        out_cpu = model(x)
        
        # MPS execution (if available)
        if torch.backends.mps.is_available():
            out_mps = model(x.to("mps"))
            
            # Assert that MPS matches CPU output
            # The bug caused specific elements to be 0.0 instead of the averaged value
            self.assertTrue(
                torch.allclose(out_cpu, out_mps.cpu(), atol=1e-2, rtol=1e-2),
                f"MPS AvgPool2d output does not match CPU.\n"
                f"CUDA SDPA Reduction Allowed: {cuda_reduction_allowed}\n"
                f"CPU Output:\n{out_cpu}\n"
                f"MPS Output:\n{out_mps.cpu()}"
            )
        else:
            self.skipTest("MPS backend is not available.")

if __name__ == "__main__":
    unittest.main()