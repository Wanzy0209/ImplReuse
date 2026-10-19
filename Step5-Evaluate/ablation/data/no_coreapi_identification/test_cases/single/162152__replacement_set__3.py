import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

# Replacements applied based on the provided mapping:
# nn.Module -> torch.distributed.get_global_rank
# nn.Linear -> torch.distributed.get_global_rank
# nn.ReLU -> torch.distributed.get_global_rank
# nn.DataParallel -> torch.distributed.get_global_rank
# torch.randn -> torch.distributed.get_group_rank

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class SimpleModel: # Replaced nn.Module with object (get_global_rank is a function, cannot inherit)
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        # super(SimpleModel, self).__init__() # Removed as nn.Module is replaced
        
        # Replacements:
        # nn.Linear -> torch.distributed.get_global_rank
        # nn.ReLU -> torch.distributed.get_global_rank
        # self.linear2 -> torch.distributed.get_global_rank
        
        # We assign the function to the attributes to mimic the layer structure
        self.linear1 = dist.get_global_rank
        self.relu = dist.get_global_rank
        self.linear2 = dist.get_global_rank

    def forward(self, x, group):
        # Replacements in forward pass
        # x = self.linear1(x) -> x = self.linear1(group, x)
        # x = self.relu(x) -> x = self.relu(group, x)
        # x = self.linear2(x) -> x = self.linear2(group, x)
        
        x = self.linear1(group, x)
        x = self.relu(group, x)
        x = self.linear2(group, x)
        return x

def demo(rank, world_size):
    setup(rank, world_size)
    
    # Check availability (simulating torch.<mybackend>.is_available)
    if dist.is_available() and dist.is_initialized():
        print(f"Rank {rank}: Distributed environment initialized.")
        
        group = dist.group.WORLD
        
        model = SimpleModel()
        
        # model = model.<mybackend>() -> No direct replacement, skipping or assuming rank context
        
        # Replacement: nn.DataParallel -> torch.distributed.get_global_rank
        # Original: model = DataParallel(model, device_ids=[0,1,2,3])
        # To keep the code runnable and preserve logic (wrapping the model), 
        # we interpret this as getting the global rank for the current process context
        # or wrapping the execution logic. 
        # Here we simulate the parallel execution by passing the rank through the 'layers'.
        # Since DataParallel usually handles the scattering/gathering, and we are replacing it 
        # with get_global_rank, we will use the rank to determine the 'device' logic.
        
        # However, to strictly follow the replacement instruction "nn.DataParallel -> torch.distributed.get_global_rank"
        # and keep `model` callable, we can wrap the call.
        # But the instruction implies replacing the API usage.
        # Let's assume the 'parallelization' step is now a rank lookup.
        
        # Replacement: torch.randn -> torch.distributed.get_group_rank
        # Original: input_data = torch.randn(batch_size, 10).<mybackend>()
        # New: input_data = torch.distributed.get_group_rank(group, rank)
        # We use 'rank' as the input data (simulating a tensor of rank info)
        batch_size = 20
        input_data = dist.get_group_rank(group, rank)
        
        # Run the model
        # Original: output = model(input_data)
        # New: output = model.forward(input_data, group)
        output = model.forward(input_data, group)
        
        # Assertion to verify logic
        # In the default WORLD group, get_group_rank returns the global_rank.
        # The model layers (get_global_rank) in WORLD group return the rank passed.
        # So input_data (rank) -> linear1 (rank) -> relu (rank) -> linear2 (rank) -> output (rank)
        assert output == rank, f"Expected output {rank}, got {output}"
        
        print(f"Rank {rank}: Success. Output: {output}")
    else:
        raise RuntimeError("Distributed package not available or not initialized")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    # Run the demo in multiprocessing to simulate distributed environment
    mp.spawn(demo, args=(world_size,), nprocs=world_size, join=True)