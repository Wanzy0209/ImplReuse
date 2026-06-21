import torch
import unittest

class TestCompileComboKernels(unittest.TestCase):
    def test_cumsum_combo_kernels_with_flash_sdp(self):
        """
        Test that torch.compile works with combo_kernels enabled and cumsum operations,
        while checking the state of the similar API torch.backends.cuda.flash_sdp_enabled.
        
        This test reproduces the scenario from Issue 162756 where a NameError was raised
        for Triton helper functions when combo_kernels was True.
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

        # Check if torch._inductor is available, as it is a private API
        # and might not be present in all PyTorch versions or builds.
        if not hasattr(torch, '_inductor'):
            self.skipTest("torch._inductor is not available in this environment")

        # Enable the configuration that triggers the original bug
        torch._inductor.config.combo_kernels = True

        # Leverage the similar API to check the backend configuration state.
        # This mirrors the pattern of accessing backend flags seen in the issue.
        is_flash_sdp_enabled = torch.backends.cuda.flash_sdp_enabled()

        @torch.compile
        def fn(x, y, z):
            # cumsum requires helper functions (e.g., _triton_helper_fn_add0)
            # which were the source of the NameError in the original bug.
            return x.sum(1), y.mean(1), z.cumsum(1)

        inps = (
            torch.rand(16, 128, device="cuda"),
            torch.rand(32, 128, device="cuda"),
            torch.rand(32, 256, device="cuda"),
        )

        try:
            # Run the compiled function. 
            # If the bug is present, this will raise:
            # NameError: '_triton_helper_fn_add0 is not defined'
            result = fn(*inps)

            # Assertions to verify correct execution
            self.assertEqual(result[0].shape, (16,))
            self.assertEqual(result[1].shape, (32,))
            self.assertEqual(result[2].shape, (32, 256))
            
            # Verify the result values are not NaN (basic sanity check)
            self.assertFalse(torch.isnan(result[0]).any())
            self.assertFalse(torch.isnan(result[1]).any())
            self.assertFalse(torch.isnan(result[2]).any())

        except NameError as e:
            self.fail(f"NameError raised during compilation: {e}")
        finally:
            # Reset configuration to avoid side effects
            if hasattr(torch, '_inductor'):
                torch._inductor.config.combo_kernels = False

if __name__ == "__main__":
    unittest.main()