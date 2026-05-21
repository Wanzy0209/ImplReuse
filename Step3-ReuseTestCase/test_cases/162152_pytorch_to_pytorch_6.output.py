import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import os

class SimpleModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize process group using gloo backend for CPU compatibility
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def demo_ddp(rank, world_size):
    setup(rank, world_size)

    # Create model and move it to the appropriate device
    # Note: For this minimal test, we use CPU. For GPU, use .to(rank) and ensure device_ids is set.
    model = SimpleModel().to(rank)

    # Adaptation: Replace DataParallel with DistributedDataParallel
    # DDP requires the model to be on the device before wrapping
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank] if torch.cuda.is_available() else None)

    batch_size = 20
    # Create input data on the specific device
    input_data = torch.randn(batch_size, 10).to(rank)

    output = model(input_data)
    
    # Verify output shape
    assert output.shape == (batch_size, 5), f"Expected shape (20, 5), got {output.shape}"
    print(f"Rank {rank}: success")

    cleanup()

def main():
    world_size = 2
    # Use spawn to launch processes for DDP
    mp.spawn(demo_ddp,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    main()