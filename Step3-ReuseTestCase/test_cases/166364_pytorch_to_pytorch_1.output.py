import torch
import torch.nn as nn
import torch.distributed as dist
import os

def test_reduce_with_learnable_scalar():
    """
    Test function to verify torch.distributed.reduce works with learnable scalars.
    Adapted from Issue 166364 regarding Flex Attention and learnable scalars.
    """
    # Initialize distributed environment
    if not dist.is_available():
        print("torch.distributed not available. Skipping test.")
        return

    # Initialize process group if not already initialized
    # In a real test harness, this is handled externally, but for a standalone script:
    if not dist.is_initialized():
        try:
            # Using TCP store for single-node testing
            dist.init_process_group(
                backend="gloo",
                init_method="tcp://127.0.0.1:29500",
                rank=0,
                world_size=1
            )
        except Exception as e:
            print(f"Failed to initialize process group: {e}")
            return

    print("Testing torch.distributed.reduce with learnable scalar...")

    # Case 1: Learnable Scalar (The problematic case in the original bug)
    try:
        temp = nn.Parameter(torch.tensor(0.0))
        # Attempt to reduce the scalar parameter
        # In the original bug, this caused issues with vmap/compile.
        # Here we check if reduce handles the scalar parameter type.
        dist.reduce(temp, dst=0)
        print("Case 1 (Scalar): Passed")
    except Exception as e:
        print(f"Case 1 (Scalar): Failed with error: {e}")

    # Case 2: Batched Parameter (The workaround in the original bug)
    try:
        B = 4
        temp_batched = nn.Parameter(torch.randn(B))
        dist.reduce(temp_batched, dst=0)
        print("Case 2 (Batched): Passed")
    except Exception as e:
        print(f"Case 2 (Batched): Failed with error: {e}")

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    test_reduce_with_learnable_scalar()