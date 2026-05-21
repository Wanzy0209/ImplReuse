import unittest
import time
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.tensor import distribute_tensor, Shard
from torch.distributed.device_mesh import init_device_mesh

def run_test(rank, world_size):
    """
    Worker function to reproduce the performance issue.
    This logic mirrors the original bug report but is structured for a test environment.
    """
    # Initialize process group
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{29500}", # Free port assumption
        rank=rank,
        world_size=world_size
    )

    # Setup device mesh
    device_mesh = init_device_mesh("cpu", (world_size,))

    # Create a large number of tensors to trigger the O(n^2) complexity
    # Using 256 tensors for the test to keep runtime reasonable, 
    # but large enough to show the quadratic behavior if the bug exists.
    num_tensors = 256
    fully_tensor = [torch.ones(1024, 1) for _ in range(num_tensors)]
    sharded_tensor = [
        distribute_tensor(tensor=t, device_mesh=device_mesh, placements=[Shard(0)]) 
        for t in fully_tensor
    ]
    state_dict = {str(key): value for key, value in enumerate(sharded_tensor)}

    planner = DefaultSavePlanner()
    planner.set_up_planner(state_dict=state_dict, is_coordinator=rank == 0)
    local_plan = planner.create_local_plan()
    
    gather_objs = [None] * world_size
    dist.gather_object(
        obj=local_plan, 
        object_gather_list=gather_objs if rank == 0 else None, 
        dst=0, 
        group=dist.group.WORLD
    )

    if rank == 0:
        start = time.time()
        # This is the call that is slow in the bug report
        all_local_plans, global_metadata = planner.create_global_plan(gather_objs)
        end = time.time()
        duration = end - start
        
        print(f"Create global planner cost {duration:.4f}s")
        
        # Assertion to verify the fix.
        # With the bug (O(n^2)), 256 tensors might take significant time (seconds).
        # With the fix, it should be very fast (milliseconds).
        # We set a generous threshold (e.g., 2.0 seconds) to avoid flakiness 
        # but catch the severe regression.
        assert duration < 2.0, f"Validation took {duration}s, expected < 2.0s. Bug might be present."

    dist.barrier()
    dist.destroy_process_group()

class TestDefaultSavePlannerPerformance(unittest.TestCase):
    """
    Test case for Issue 163548: DefaultSavePlanner._validate_global_plan performance.
    
    This test verifies that creating a global plan does not exhibit O(n^2) complexity.
    It leverages the concept of handling long-running operations (similar to 
    tf.errors.CancelledError scenarios) by enforcing a strict timeout/assertion 
    on the execution time.
    """
    
    def setUp(self):
        self.world_size = 2 # Minimal size to test distributed logic
        
    def test_global_plan_performance(self):
        """
        Spawns processes to run the performance test.
        """
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")
        p = ctx.Process(
            target=run_test, 
            args=(0, self.world_size)
        )
        p.start()
        p.join(timeout=30) # Overall test timeout
        
        if p.is_alive():
            p.terminate()
            self.fail("Test timed out, likely due to the O(n^2) performance bug.")
        elif p.exitcode != 0:
            self.fail(f"Worker process failed with exit code {p.exitcode}")

if __name__ == "__main__":
    # Note: This test requires torch.distributed to be initialized properly.
    # Running directly might fail if not launched with torchrun or similar,
    # but the logic inside run_test handles the init.
    # For simplicity in this snippet, we assume the environment supports multiprocessing.
    unittest.main()