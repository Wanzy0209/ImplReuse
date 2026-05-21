import torch
import unittest

class TestMPSCompileContiguity(unittest.TestCase):
    def test_mps_compile_cholesky_inverse_vmap(self):
        """
        Test case for Issue 161969.
        Verifies that torch.compile works correctly on MPS with operations
        involving cholesky, inverse, and vmap/grad without triggering
        contiguity assertion failures.
        
        This test leverages the pattern from tf.test.is_built_with_xla to 
        check for backend availability before running the compiled test.
        """
        # Leverage the pattern of checking backend availability (similar to tf.test.is_built_with_xla)
        # to ensure the test only runs on supported hardware.
        if not torch.backends.mps.is_available():
            self.skipTest("MPS is not available, skipping test.")

        device = torch.device("mps")
        dtype = torch.float32

        def logp(x, matrix):
            # The original bug report noted that adding a print statement here 
            # (print(matrix.is_contiguous())) would make the code run.
            # This test verifies that the code runs correctly without that workaround.
            p_mat_sqrt = torch.linalg.cholesky(matrix).contiguous()
            p_mat_sqrt_inv = p_mat_sqrt.inverse()
            val = torch.sum((p_mat_sqrt_inv @ x[0, :]) ** 2)
            return -val / 2

        # Wrap in vmap and grad as per the bug report
        score_func = torch.vmap(torch.func.grad(logp, 0), (0, None))

        # Compile the function
        compiled_function = torch.compile(score_func)

        # Setup inputs
        data = torch.zeros((2, 5, 3), device=device, dtype=dtype)
        p = torch.diag(torch.tensor((20., 0.5, 5.), device=device, dtype=dtype)**2)

        # Execute and verify
        # Expectation: No RuntimeError regarding contiguity
        try:
            res = compiled_function(data, p)
            self.assertIsNotNone(res)
            self.assertEqual(res.shape, (2, 5, 3))
        except RuntimeError as e:
            if "is_contiguous() INTERNAL ASSERT FAILED" in str(e):
                self.fail(f"Bug reproduced: MPS contiguity assertion failed during compile: {e}")
            raise

if __name__ == "__main__":
    unittest.main()