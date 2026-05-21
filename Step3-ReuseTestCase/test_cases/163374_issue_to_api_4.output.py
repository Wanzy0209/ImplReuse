import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed._tensor import Shard, distribute_tensor, init_device_mesh, Replicate

def test_dtensor_inplace_clamp(rank, world_size):
    """
    Test case for Issue 163374: [DTensor] Inplace ops produces wrong result.
    
    This test reproduces the logic of the bug report where an inplace operation (clamp_)
    on a partial DTensor results in incorrect placement and value.
    
    It leverages the semantics of the similar API 'tf.raw_ops.Assert' by using 
    Python assertions to verify the expected conditions (placement and value) 
    rather than just printing them.
    """
    # Initialize the distributed environment
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    
    # Initialize device mesh (using CPU for compatibility, original used CUDA)
    mesh = init_device_mesh('cpu', (world_size,))

    # Create a sharded tensor
    tensor = torch.ones(12, 12, device="cpu")
    in_dtensor = distribute_tensor(tensor, mesh, [Shard(0)]) 

    # Perform a reduction which results in a Partial placement
    partial_dt = in_dtensor.sum()
    
    # Apply the inplace operation under test
    # Bug: This operation historically failed to redistribute from Partial to Replicate
    out = partial_dt.clamp_(max=2)

    # --- Verification (Leveraging tf.raw_ops.Assert semantics) ---
    
    # Assert 1: Verify Placement
    # The bug report indicates the placement remains Partial(sum) instead of becoming Replicate().
    # We assert the expected behavior here.
    expected_placement = (Replicate(),)
    assert out.placements == expected_placement, (
        f"[Rank {rank}] Placement mismatch. Expected {expected_placement}, got {out.placements}. "
        "The inplace op did not redistribute the tensor correctly."
    )

    # Assert 2: Verify Value
    # The bug report indicates the value is 144 (sum of ones) instead of 2 (clamped).
    full = out.full_tensor()
    expected_value = torch.tensor(2.0, device="cpu")
    assert torch.equal(full, expected_value), (
        f"[Rank {rank}] Value mismatch. Expected {expected_value}, got {full}. "
        "The inplace op did not apply the clamp correctly."
    )

    dist.destroy_process_group()

if __name__ == '__main__':
    # Use multiprocessing to simulate the distributed environment
    world_size = 2
    mp.spawn(test_dtensor_inplace_clamp, args=(world_size,), nprocs=world_size, join=True)