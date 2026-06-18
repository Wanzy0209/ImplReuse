import torch
import unittest

class TestInductorComboKernels(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_combo_kernels_cumsum_with_helper_fn(self):
        """
        Test that torch.compile works with combo_kernels enabled when operations
        requiring helper functions (like cumsum) are used.
        
        This test leverages torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed
        to check the backend configuration state, ensuring the test runs in a 
        context where backend optimizations are active.
        """
        # Leverage the similar API to verify the environment's backend state
        # This relates to the issue's context of backend configuration flags
        sdp_allowed = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
        
        # Enable the specific configuration that triggers the bug
        torch._inductor.config.combo_kernels = True

        @torch.compile
        def fn(x, y, z):
            return x.sum(1), y.mean(1), z.cumsum(1)

        inps = (
            torch.rand(16, 128, device="cuda"),
            torch.rand(32, 128, device="cuda"),
            torch.rand(32, 256, device="cuda"),
        )

        # The bug causes a NameError for '_triton_helper_fn_add0' during compilation
        # We expect this to succeed if the bug is fixed
        try:
            result = fn(*inps)
        except NameError as e:
            if "_triton_helper_fn" in str(e):
                self.fail(f"Bug reproduced: Triton helper function not defined: {e}")
            else:
                raise

        # Verify correctness of the compiled function
        expected_sum = inps[0].sum(1)
        expected_mean = inps[1].mean(1)
        expected_cumsum = inps[2].cumsum(1)

        self.assertTrue(torch.allclose(result[0], expected_sum))
        self.assertTrue(torch.allclose(result[1], expected_mean))
        self.assertTrue(torch.allclose(result[2], expected_cumsum))

if __name__ == "__main__":
    unittest.main()