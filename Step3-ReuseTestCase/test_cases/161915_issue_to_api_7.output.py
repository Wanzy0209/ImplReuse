import torch
import os
import tempfile
import unittest

class TestNestedTensorShareMemory(unittest.TestCase):
    def test_nested_tensor_share_memory_with_profiling(self):
        """
        Test that share_memory_() works on NestedTensor with jagged layout
        without causing a segmentation fault.
        
        This test leverages the similar API (torch.profiler.tensorboard_trace_handler)
        to wrap the execution, ensuring the memory sharing operation is stable
        even when traced/profiled, and reuses the directory handling pattern
        found in the similar API's implementation.
        """
        # Setup a temporary directory for the trace handler, 
        # mimicking the directory creation logic in the similar API
        with tempfile.TemporaryDirectory() as log_dir:
            # Reuse the similar API: torch.profiler.tensorboard_trace_handler
            trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)

            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                on_trace_ready=trace_handler
            ) as prof:
                # --- Original Bug Reproduction Logic ---
                a = torch.randn(3)
                b = torch.randn(5)
                nt = torch.nested.nested_tensor([a, b], layout=torch.jagged)
                
                # This call previously caused a Segmentation fault (core dumped)
                nt.share_memory_()
                # ---------------------------------------

                # Verify the operation succeeded and the tensor is actually shared
                self.assertTrue(nt.is_shared(), "NestedTensor should be in shared memory after calling share_memory_()")

            # Verify that the trace handler successfully wrote a file
            # (This confirms the similar API integration worked as expected)
            trace_files = os.listdir(log_dir)
            self.assertTrue(len(trace_files) > 0, "Trace handler should have generated a trace file")

if __name__ == '__main__':
    unittest.main()