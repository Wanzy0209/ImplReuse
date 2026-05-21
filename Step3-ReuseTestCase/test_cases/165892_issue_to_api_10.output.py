import torch
import tempfile
import unittest
import os

class TestBmmCompileWithProfiler(unittest.TestCase):
    def test_bmm_out_dtype_compile_with_trace_handler(self):
        """
        Test case for Issue #165892: torch.bmm + torch.compile with out_dtype.
        Leverages torch.profiler.tensorboard_trace_handler to profile the execution,
        ensuring the compiled kernel runs correctly under inspection.
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

        # Setup inputs as per the bug report
        A = torch.rand((1, 1024, 1024), device="cuda", dtype=torch.float16)
        B = torch.rand((1, 1024, 1024), device="cuda", dtype=torch.float16)

        # The function causing the regression in 2.9
        @torch.compile
        def linear(weight, input):
            return torch.bmm(input, weight, out_dtype=torch.float32)

        # Leverage the similar API: torch.profiler.tensorboard_trace_handler
        # We use a temporary directory to store the trace output
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize the trace handler
            trace_handler = torch.profiler.tensorboard_trace_handler(tmpdir)

            # Run the compiled function within a profiler context using the handler
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                on_trace_ready=trace_handler
            ) as p:
                # This call previously raised:
                # torch._inductor.exc.InductorError: AssertionError: out_dtype is not supported for Triton
                result = linear(A, B)

            # Verify the trace file was created (sanity check for the handler)
            trace_files = os.listdir(tmpdir)
            self.assertTrue(len(trace_files) > 0, "Profiler trace file was not created")

        # Verify the output correctness
        self.assertEqual(result.shape, (1, 1024, 1024))
        self.assertEqual(result.dtype, torch.float32)

if __name__ == "__main__":
    unittest.main()