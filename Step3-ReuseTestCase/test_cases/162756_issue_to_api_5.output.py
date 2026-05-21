import torch
import unittest

class TestCompileComboKernels(unittest.TestCase):
    def test_compile_combo_kernels_with_cumsum(self):
        # Leverage the similar API to ensure the CUDA environment is capable
        # of running the specific Triton operations required for the test.
        if not torch.backends.cuda.is_flash_attention_available():
            self.skipTest("Flash Attention (and thus CUDA/Triton) not available")

        # Enable the configuration that triggers the bug
        torch._inductor.config.combo_kernels = True

        @torch.compile
        def fn(x, y, z):
            # cumsum requires helper functions (e.g., associative_scan)
            # which were causing NameError in combo kernels
            return x.sum(1), y.mean(1), z.cumsum(1)

        inps = (
            torch.rand(16, 128, device="cuda"),
            torch.rand(32, 128, device="cuda"),
            torch.rand(32, 256, device="cuda"),
        )

        # Run the function. If the bug is present, this will raise
        # torch._dynamo.exc.BackendCompilerFailed: NameError('_triton_helper_fn_add0 is not defined')
        try:
            res = fn(*inps)
        except Exception as e:
            self.fail(f"torch.compile failed with combo_kernels enabled: {e}")

        # Basic assertions to verify correctness
        self.assertEqual(res[0].shape, (16,))
        self.assertEqual(res[1].shape, (32,))
        self.assertEqual(res[2].shape, (32, 256))

if __name__ == "__main__":
    unittest.main()