import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

# This class mimics the structure of tf.keras.initializers.Identity
# (Class-based initialization and execution via __call__) to encapsulate
# the distributed process group lifecycle.
class DistributedProcessInitializer:
    def __init__(self, rank, world_size, backend="nccl"):
        """
        Initialize the distributed process runner.
        Analogous to tf.keras.initializers.Identity.__init__
        """
        self.rank = rank
        self.world_size = world_size
        self.backend = backend

    def __call__(self, shape=None, dtype=None, **kwargs):
        """
        Executes the distributed process logic.
        Analogous to tf.keras.initializers.Identity.__call__
        """
        # Set device
        torch.cuda.set_device(self.rank)
        
        # Initialize process group
        dist.init_process_group(
            backend=self.backend,
            init_method="tcp://127.0.0.1:29500",
            world_size=self.world_size,
            rank=self.rank
        )
        
        # Synchronize processes
        dist.barrier()
        
        # Perform some work to ensure CUDA context is active
        device = f"cuda:{self.rank}"
        # Creating a tensor to simulate workload
        _ = torch.randn(1024, 1024, device=device)
        
        # Destroy the process group
        dist.destroy_process_group()
        
        # Sleep to allow observation of memory state (mimicking original bug report)
        time.sleep(2)
        
        # Check memory status after destruction
        # The bug reports unexpected 320MB allocation persisting here.
        torch.cuda.empty_cache()
        allocated_mem = torch.cuda.memory_allocated(self.rank)
        reserved_mem = torch.cuda.memory_reserved(self.rank)
        
        print(f"Rank {self.rank} - After destroy: Allocated={allocated_mem}, Reserved={reserved_mem}")

def worker(rank, world_size):
    # Instantiate the class similar to how one would use a Keras Initializer
    process_runner = DistributedProcessInitializer(rank, world_size)
    # Execute the logic
    process_runner()

def test_distributed_memory_leak():
    """
    Test case to reproduce unexpected CUDA memory allocation 
    after dist.destroy_process_group.
    """
    world_size = torch.cuda.device_count()
    if world_size < 1:
        print("Test requires at least 1 GPU.")
        return

    mp.spawn(
        worker,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )

if __name__ == '__main__':
    test_distributed_memory_leak()