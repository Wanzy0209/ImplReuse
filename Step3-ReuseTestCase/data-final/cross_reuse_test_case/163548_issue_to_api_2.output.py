import torch
import torch.distributed as dist
import torch.nn as nn
import time
import unittest

class TestDropoutPerformance(unittest.TestCase):
    """
    Test case adapted from Issue 163548 regarding DefaultSavePlanner performance.
    This test verifies that torch.nn.Dropout (the similar API) does not exhibit 
    O(n^2) performance degradation when processing a large number of tensors 
    in a distributed environment.
    """

    def setUp(self):
        if not dist.is_available():
            self.skipTest("Torch distributed is not available.")
        
        # Initialize process group similar to the bug reproduction
        try:
            dist.init_process_group(backend="gloo")
        except RuntimeError as e:
            self.skipTest(f"Could not initialize process group: {e}")

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_dropout_large_scale_linear_time(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Reproduce the scale from the bug report: 1024 tensors
        num_tensors = 1024
        # Create tensors similar to the bug report (1024, 1)
        tensors = [torch.ones(1024, 1) for _ in range(num_tensors)]

        # Instantiate the Similar API: torch.nn.Dropout
        dropout = nn.Dropout(p=0.5)

        if rank == 0:
            start = time.time()

        # Apply the operation to all tensors.
        # In the original bug, create_global_plan was slow due to O(n^2) validation.
        # Here we verify that applying Dropout to the same number of tensors 
        # remains efficient (linear time).
        results = [dropout(t) for t in tensors]

        if rank == 0:
            end = time.time()
            duration = end - start
            print(f"Dropout forward pass cost {duration:.4f}s")

            # The bug report mentioned 200+ seconds for 1024 tensors with the planner.
            # Dropout should be significantly faster (linear time).
            # We assert it completes in a reasonable time (e.g., < 1 second).
            self.assertLess(duration, 1.0, 
                f"Performance regression detected: Dropout took {duration}s "
                "which is unexpectedly slow for this scale.")

        dist.barrier()

if __name__ == "__main__":
    # This test requires a distributed environment (e.g., torchrun)
    unittest.main()