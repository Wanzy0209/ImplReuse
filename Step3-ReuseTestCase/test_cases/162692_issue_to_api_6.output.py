import torch
import torch.distributed as dist
from torch.distributed._tensor import Shard, distribute_tensor, init_device_mesh
from torch import nn

def test_dtensor_mean_uneven_sharding():
    """
    Test case for Issue #162692: Incorrect results of DTensor.mean with uneven sharding.
    
    This test verifies that DTensor.mean correctly handles uneven sharding scenarios.
    It leverages the similar API pattern (HeUniform/Kaiming Uniform) to initialize 
    the tensor data, ensuring the test covers realistic weight initialization scenarios 
    where uneven sharding might occur.
    """
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    mesh = init_device_mesh('cuda', (2,))

    # Case 1: 3x4 tensor (Uneven sharding: Rank 0 gets 2 rows, Rank 1 gets 1 row)
    # Using Kaiming Uniform (PyTorch equivalent of HeUniform) for initialization
    tensor_3x4 = torch.empty(3, 4, device='cuda')
    nn.init.kaiming_uniform_(tensor_3x4, a=2)  # HeUniform uses a=2 (mode='fan_in')
    
    truth_3x4 = tensor_3x4.mean()
    dt_3x4 = distribute_tensor(tensor_3x4, device_mesh=mesh, placements=[Shard(0)])
    
    mean_3x4 = dt_3x4.mean()
    full_3x4 = mean_3x4.full_tensor()
    
    # Assert that the distributed mean matches the local mean
    assert torch.allclose(full_3x4, truth_3x4, atol=1e-5), \
        f"Rank {rank}: Mean mismatch for 3x4 tensor. Got {full_3x4}, expected {truth_3x4}"

    # Case 2: 1x4 tensor (Uneven sharding: Rank 0 gets 1 row, Rank 1 gets 0 rows)
    # This case resulted in NaN in the original bug report
    tensor_1x4 = torch.empty(1, 4, device='cuda')
    nn.init.kaiming_uniform_(tensor_1x4, a=2)
    
    truth_1x4 = tensor_1x4.mean()
    dt_1x4 = distribute_tensor(tensor_1x4, device_mesh=mesh, placements=[Shard(0)])
    
    mean_1x4 = dt_1x4.mean()
    full_1x4 = mean_1x4.full_tensor()
    
    # Assert that the result is not NaN and matches the truth
    assert not torch.isnan(full_1x4), \
        f"Rank {rank}: Mean is NaN for 1x4 tensor"
    assert torch.allclose(full_1x4, truth_1x4, atol=1e-5), \
        f"Rank {rank}: Mean mismatch for 1x4 tensor. Got {full_1x4}, expected {truth_1x4}"

    print(f"Rank {rank}: Test passed.")

if __name__ == '__main__':
    test_dtensor_mean_uneven_sharding()