import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

# Move worker function to module level to make it picklable for mp.spawn
def worker(rank, world_size):
    # Setup distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Create a bfloat16 tensor mimicking the logits from the original bug report context.
    # We use deterministic values based on rank to verify correctness.
    # Original context: meta-llama/Llama-3.2-1B with bfloat16.
    local_tensor = torch.ones(1, 1000, dtype=torch.bfloat16) * rank
    
    gather_list = [None] * world_size if rank == 0 else None

    # Call the API under test: torch.distributed.gather_object
    # This replaces the torch.compile call site from the original test.
    dist.gather_object(local_tensor, gather_list, dst=0)

    if rank == 0:
        # Verify the gathered objects match the expected values
        for i in range(world_size):
            expected_tensor = torch.ones(1, 1000, dtype=torch.bfloat16) * i
            # Use standard assert instead of self.assertTrue since we are outside the class
            assert torch.equal(gather_list[i], expected_tensor), f"Tensor mismatch at index {i}"
            assert gather_list[i].dtype == torch.bfloat16, f"Dtype mismatch at index {i}"

    dist.destroy_process_group()

class TestDistributedGatherObject(unittest.TestCase):
    def test_gather_object_bf16_correctness(self):
        """
        Adapted from Issue 161457: [compile] meta-llama correctness issue.
        Original issue involved bfloat16 accuracy problems with Llama models.
        This test verifies that torch.distributed.gather_object correctly
        handles bfloat16 tensors (objects) across processes, ensuring data integrity.
        """
        
        # Spawn 2 processes to simulate distributed environment
        world_size = 2
        mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    unittest.main()