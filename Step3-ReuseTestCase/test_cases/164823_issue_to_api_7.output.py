import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class TestModel(nn.Module):
    def forward(self, x):
        # Leverage the similar API: torch.distributed.all_gather_into_tensor
        # This mimics the structure of the original bug where a tensor operation
        # is performed inside the model's forward pass.
        # Note: all_gather_into_tensor requires an output tensor.
        # For a single process (world_size=1), this effectively copies input to output.
        output_tensor = torch.empty_like(x)
        dist.all_gather_into_tensor(output_tensor, x)
        return output_tensor

def run_test(rank, world_size):
    setup(rank, world_size)
    
    # Create a dummy input tensor
    x = torch.randn(10, 10)
    
    model = TestModel()
    
    # Eager execution
    print(f"Rank {rank}: Eager output shape:", model(x).shape)
    
    # Compiled execution
    # This tests if torch.compile handles the similar API (all_gather_into_tensor)
    # correctly, similar to how it failed with to_sparse().
    try:
        compiled_model = torch.compile(model)
        print(f"Rank {rank}: Compiled output shape:", compiled_model(x).shape)
        print("Test Passed: torch.compile works with all_gather_into_tensor")
    except Exception as e:
        print(f"Rank {rank}: Test Failed with error: {e}")

    cleanup()

if __name__ == "__main__":
    world_size = 1
    # Use spawn to run the distributed test
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)