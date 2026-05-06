"""Minimal reproduction to test DTensor shard propagation caching failure.

This script tests if DTensor shard propagation caching is working by:
1. Performing torch.stack on DTensors twice with the same structure
2. Monitoring calls to the non-cached propagation function
3. Printing 0 if caching works (no non-cached calls on 2nd run)
4. Printing 1 if caching fails (non-cached calls on 2nd run)

Expected behavior: First call should populate cache, second call should hit cache.
Bug: Second call is still going through non-cached propagation.
"""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, DeviceMesh, Replicate


def main():
    # Initialize distributed environment
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Monkey-patch the non-cached propagation function to count calls
    try:
        from torch.distributed.tensor._sharding_prop import ShardingPropagator

        original_propagate = ShardingPropagator.propagate_op_sharding_non_cached
        call_counts = {"first": 0, "second": 0, "phase": "first"}

        def counting_propagate(self, op_schema):
            call_counts[call_counts["phase"]] += 1
            return original_propagate(self, op_schema)

        # Apply the monkey patch at class level (before any OpDispatcher is created)
        ShardingPropagator.propagate_op_sharding_non_cached = counting_propagate

        # Create a simple device mesh
        device_mesh = DeviceMesh("cpu", torch.arange(world_size))

        # Create DTensor tensors similar to FusedPerParamGradientNormalize
        shape = (4, 8)
        placements = [Replicate()]

        # === FIRST CALL ===
        # Expected: Should go through non-cached path to populate cache
        call_counts["phase"] = "first"
        call_counts["first"] = 0

        dtensor_list = []
        for _ in range(3):
            local_tensor = torch.randn(shape)
            dt = DTensor.from_local(local_tensor, device_mesh, placements, run_check=False)
            dtensor_list.append(dt)

        result1 = torch.stack(dtensor_list)
        first_non_cached_calls = call_counts["first"]

        # === SECOND CALL ===
        # Expected: Should hit cache, NOT call non-cached propagation
        # Bug: If this calls non-cached propagation, caching is broken
        call_counts["phase"] = "second"
        call_counts["second"] = 0

        dtensor_list2 = []
        for _ in range(3):
            local_tensor = torch.randn(shape)
            dt = DTensor.from_local(local_tensor, device_mesh, placements, run_check=False)
            dtensor_list2.append(dt)

        result2 = torch.stack(dtensor_list2)
        second_non_cached_calls = call_counts["second"]

        # Restore original function
        ShardingPropagator.propagate_op_sharding_non_cached = original_propagate

        if rank == 0:
            print(f"# First call: {first_non_cached_calls} non-cached propagation(s)")
            print(f"# Second call: {second_non_cached_calls} non-cached propagation(s)")
            print(f"# Expected second call: 0 non-cached propagations (should use cache)")

        # Output result:
        # 0 = caching works (second call uses cache)
        # 1 = caching FAILS (second call still goes through non-cached path)
        if second_non_cached_calls > 0:
            print(1)  # FAIL: Caching is broken
        else:
            print(0)  # PASS: Caching works

    except Exception as e:
        print(f"# Error: {e}")
        import traceback
        traceback.print_exc()
        print(0)  # Assume it works if we can't measure

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()