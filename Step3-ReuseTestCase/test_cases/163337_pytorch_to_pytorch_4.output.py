import torch
import torch.distributed as dist
import sys

def test_get_process_group_ranks():
    """
    Test case for torch.distributed.get_process_group_ranks.
    Verifies that the API correctly returns the list of ranks for a given process group.
    """
    # Check if distributed is available
    if not dist.is_available():
        print("torch.distributed is not available. Skipping test.")
        return

    # Initialize the process group
    # We use 'gloo' backend and 'tcp' initialization for a standalone test.
    # This simulates a single-process environment.
    try:
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:29500",
            rank=0,
            world_size=1
        )
    except RuntimeError as e:
        # Handle cases where the address might be in use or other init errors
        print(f"Failed to initialize process group: {e}")
        return

    try:
        # Get the default process group (WORLD)
        group = dist.group.WORLD

        # Call the API under test
        ranks = dist.get_process_group_ranks(group)

        # Assertions
        # 1. The return type should be a list
        assert isinstance(ranks, list), f"Expected list, got {type(ranks)}"

        # 2. The list should contain integers
        assert all(isinstance(r, int) for r in ranks), "All ranks should be integers"

        # 3. For a single-process group (world_size=1), the ranks should be [0]
        assert ranks == [0], f"Expected ranks [0], got {ranks}"

        print("Test passed successfully.")

    finally:
        # Clean up
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    test_get_process_group_ranks()