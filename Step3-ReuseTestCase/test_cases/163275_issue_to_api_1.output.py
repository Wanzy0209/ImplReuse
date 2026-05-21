import torch
import unittest

class TestTorchCompileMmOutDtype(unittest.TestCase):
    """
    Test case to verify that torch.compile correctly handles the out_dtype argument
    for torch.mm, addressing the issue where meta_mm() failed to process the argument.
    
    This test leverages the pattern from the similar API (tf.test.Benchmark) by
    separating the operation definition (builder_fn) from the execution logic,
    allowing verification of both eager and compiled modes.
    """

    def setUp(self):
        # The original bug report specifically uses CUDA
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available. This test is based on a CUDA-specific bug report.")
        
        # Initialize inputs matching the bug report dimensions and types
        self.A = torch.rand((1024, 1024), device="cuda", dtype=torch.float16)
        self.B = torch.rand((1024, 1024), device="cuda", dtype=torch.float16)

    def _execute_mm(self, use_compile):
        """
        Helper function to execute torch.mm with out_dtype.
        Mirrors the builder_fn pattern in the similar API (tf.test.Benchmark).
        
        Args:
            use_compile (bool): If True, wraps the function with torch.compile.
                                Analogous to use_xla_jit in the similar API.
        """
        def mm_op(input, weight):
            return torch.mm(input, weight, out_dtype=torch.float32)

        if use_compile:
            mm_op = torch.compile(mm_op)
            
        return mm_op(self.A, self.B)

    def test_mm_out_dtype_compiled(self):
        """
        Test that torch.compile handles out_dtype argument in torch.mm.
        Verifies that the compiled version produces the correct dtype and
        matches the eager execution result.
        """
        # Run with compilation (use_xla_jit=True equivalent)
        result_compiled = self._execute_mm(use_compile=True)
        
        # Run without compilation (use_xla_jit=False equivalent)
        result_eager = self._execute_mm(use_compile=False)
        
        # 1. Verify the output dtype is float32 as requested
        self.assertEqual(result_compiled.dtype, torch.float32, 
                         "Compiled torch.mm did not respect out_dtype argument")
        
        # 2. Verify consistency between eager and compiled modes
        self.assertTrue(torch.allclose(result_eager, result_compiled), 
                        "Compiled output differs from eager output")

if __name__ == "__main__":
    unittest.main()