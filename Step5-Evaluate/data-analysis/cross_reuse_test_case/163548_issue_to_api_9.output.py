import torch
import torch.distributed as dist
from torch.distributed.tensor import distribute_tensor, Shard
from torch.distributed.device_mesh import init_device_mesh
import time
import sys

try:
    from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
except ModuleNotFoundError:
    print("Skipping test: 'torch.distributed.checkpoint' module not found.")
    sys.exit(0)

def test_planner_performance_with_bessel_data():
    """
    Test case for Issue 163548: DefaultSavePlanner._validate_global_plan is costly slow.
    
    This test preserves the original bug reproduction logic (checking performance of 
    create_global_plan with a large state dict) but leverages torch.special.bessel_j0 
    to generate the tensor data, satisfying the requirement to reuse the similar API.
    """
    # Initialize process group
    pg = torch.distributed.init_process_group(backend="gloo")
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)
    device_mesh = init_device_mesh("cpu", (world_size,))

    # Generate data using the similar API: torch.special.bessel_j0
    # We create a base tensor and apply the Bessel function to it.
    # This replaces the simple torch.ones from the original reproduction.
    base_input = torch.linspace(0, 10, 1024).reshape(1024, 1)
    processed_data = torch.special.bessel_j0(base_input)

    # Create a large list of tensors (1024 tensors) to trigger the O(n^2) behavior
    # in the original bug.
    fully_tensor = [processed_data.clone() for _ in range(1024)]
    
    # Distribute tensors
    sharded_tensor = [
        distribute_tensor(tensor=t, device_mesh=device_mesh, placements=[Shard(0)]) 
        for t in fully_tensor
    ]
    state_dict = {str(key): value for key, value in enumerate(sharded_tensor)}

    # Setup Planner
    planner = DefaultSavePlanner()
    planner.set_up_planner(state_dict=state_dict, is_coordinator=rank==0)
    local_plan = planner.create_local_plan()
    
    gather_objs = [None] * world_size
    dist.gather_object(
        obj=local_plan, 
        object_gather_list=gather_objs if rank == 0 else None, 
        dst=0, 
        group=pg
    )

    # Measure performance of the problematic function
    if rank == 0:
        start = time.time()
        # This call internally invokes _validate_global_plan which was the bottleneck.
        all_local_plans, global_metadata = planner.create_global_plan(gather_objs)
        end = time.time()
        duration = end - start
        
        print(f"Create global planner cost {duration:.4f}s")

        # Assertion to verify the fix.
        # The bug report mentioned 200+ seconds for 1024 tensors.
        # We expect the fix to bring this down to a reasonable time (e.g., < 5 seconds).
        assert duration < 5.0, (
            f"Validation took too long ({duration}s), indicating potential O(n^2) regression. "
            "Expected < 5.0s for this input size."
        )

    dist.barrier()

if __name__ == "__main__":
    test_planner_performance_with_bessel_data()