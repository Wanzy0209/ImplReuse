import torch
import torch.distributed as dist
import os

def setup():
    """
    Initialize the distributed process group for testing.
    Uses a single-process setup (world_size=1) to keep the test minimal and runnable.
    """
    if not dist.is_initialized():
        # Use TCP for initialization to avoid file system permissions issues
        dist.init_process_group(
            backend='gloo',
            init_method='tcp://127.0.0.1:29500',
            rank=0,
            world_size=1
        )

def process_group_creation(ranks):
    """
    Mimics the 'process' function from the original bug report.
    Instead of exporting a model, we create a new distributed group.
    """
    print(f'Creating group with ranks: {ranks}')
    # Adapted call site: torch.distributed.new_group
    group = dist.new_group(ranks=ranks)
    print(f'Group created: {group}')
    return group

if __name__ == "__main__":
    setup()

    # Test Case 1: Create group with ranks=None (default behavior)
    # Corresponds to AnyDimsModelNull in the original bug
    print("Processing first group (ranks=None)...")
    group1 = process_group_creation(ranks=None)

    # Test Case 2: Create group with specific ranks
    # Corresponds to AnyDimsModelEmpty in the original bug.
    # The original bug manifested when processing the second model sequentially.
    print("Processing second group (ranks=[0])...")
    group2 = process_group_creation(ranks=[0])

    # Verification
    assert group1 is not None, "First group creation failed"
    assert group2 is not None, "Second group creation failed"
    assert group1 != group2, "Groups should be distinct instances"

    print("Test passed: Sequential group creation handled correctly.")