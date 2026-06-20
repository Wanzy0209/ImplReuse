import torch
import unittest

# Reusing the pattern from tf.test.is_built_with_rocm to check for backend availability
def is_mps_available():
    """Returns whether PyTorch supports MPS (Apple Silicon GPU)."""
    return torch.backends.mps.is_available()

class TestMPSCompileContiguity(unittest.TestCase):
    def test_compile_vmap_grad_cholesky_mps(self):
        # Leverage the similar API pattern: Skip if hardware support is missing
        if not is_mps_available():
            self.skipTest("Test is only applicable on MPS (Apple Silicon)")

        device = torch.device("mps")
        dtype = torch.float32

        # Original bug reproduction logic
        def example_function():
            def logp(x, matrix):
                # The bug report notes that uncommenting the print below fixes the crash.
                # We leave it commented to test the actual compilation behavior.
                # print(matrix.is_contiguous())
                
                p_mat_sqrt = torch.linalg.cholesky(matrix).contiguous()
                p_mat_sqrt_inv = p_mat_sqrt.inverse()
                val = torch.sum((p_mat_sqrt_inv @ x[0, :]) ** 2)
                return -val/2

            score_func = torch.vmap(torch.func.grad(logp, 0), (0, None))
            return score_func

        data = torch.zeros((2, 5, 3), device=device, dtype=dtype)
        
        # The bug is triggered specifically by torch.compile
        compiled_function = torch.compile(example_function())

        p = torch.diag(torch.tensor((20., 0.5, 5,), device=device, dtype=dtype)**2)
        
        # Execute the compiled function
        # If the bug exists, this will raise RuntimeError: A_t.is_contiguous() INTERNAL ASSERT FAILED
        res = compiled_function(data, p)
        
        # Verify output shape to ensure execution completed
        self.assertEqual(res.shape, (2, 5, 3))

if __name__ == "__main__":
    unittest.main()