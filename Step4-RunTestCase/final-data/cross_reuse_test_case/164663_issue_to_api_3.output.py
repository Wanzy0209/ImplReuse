import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

# Handle missing dependencies for older PyTorch versions
try:
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed._composable.fsdp import fully_shard
    from torch.distributed.tensor.experimental import implicit_replication
    from torch.distributed._tools.fsdp2_mem_tracker import FSDPMemTracker
except ImportError as e:
    print(f"Skipping test: Required modules not found. This test requires PyTorch 2.x or newer. Error: {e}")
    sys.exit(0)


class TestModule(nn.Module):
    """
    Model using nn.RMSNorm wrapped with Linear.
    This structure is required to reproduce the KeyError in FSDPMemTracker.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.output(x)
        return x


def test_fsdp_mem_tracker_rmsnorm(rank, world_size):
    """
    Test case to reproduce Issue 164663.
    
    This test verifies that FSDPMemTracker does not raise a KeyError during 
    the backward pass when tracking a model containing nn.RMSNorm that is 
    sharded using fully_shard.
    
    The pattern mirrors the usage of tf.distribute.experimental.MultiWorkerMirroredStrategy,
    where a strategy scope (tracker context) wraps the execution of the distributed step.
    """
    # Setup distributed environment
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group(
        backend="nccl", 
        init_method=f"tcp://localhost:29500", 
        world_size=world_size, 
        rank=rank
    )
    torch.cuda.set_device(rank)

    d_model = 128

    # Initialize Model
    model = TestModule(d_model).cuda()
    
    # Initialize Device Mesh (similar to cluster_resolver in TF)
    mesh = init_device_mesh("cuda", (world_size,))

    # Apply Sharding
    # The bug specifically occurs when RMSNorm is wrapped with another Linear module
    # in a specific sharding configuration.
    fully_shard([model.norm, model.output], mesh=mesh)
    fully_shard(model, mesh=mesh)

    # Initialize Memory Tracker
    # This acts as the distributed execution scope, similar to strategy.scope() in TF
    tracker = FSDPMemTracker(model)

    # Run Forward and Backward pass
    # The KeyError was reported to happen during pre-backward/backward
    try:
        with tracker, implicit_replication():
            x = torch.randn(16, d_model, device='cuda')
            y = model(x)
            loss = y.sum()
            loss.backward()
        
        # If we reach here, the bug is fixed/present
        if rank == 0:
            print("Test Passed: FSDPMemTracker handled RMSNorm backward pass successfully.")
            
    except KeyError as e:
        if rank == 0:
            print(f"Test Failed: KeyError raised during backward - {e}")
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    # Run with 2 GPUs to simulate a distributed environment
    world_size = 2
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
    else:
        mp.spawn(test_fsdp_mem_tracker_rmsnorm, args=(world_size,), nprocs=world_size)