import torch
import torch.distributed as dist
import torch.nn as nn
import os

def setup():
    # Minimal setup to allow all_gather_into_tensor to run in a single-process test
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend='gloo', rank=0, world_size=1)

class TestModel(nn.Module):
    def forward(self, x):
        # Replacing x.to_sparse() logic with distributed gather logic
        # We simulate the transformation of the tensor x
        world_size = dist.get_world_size()
        
        # Prepare output buffer for all_gather_into_tensor
        # This replaces the intermediate sparse tensor storage
        output = torch.zeros(x.numel() * world_size, dtype=x.dtype, device=x.device)
        
        # Perform the collective operation
        # This replaces the sparse computation
        dist.all_gather_into_tensor(output, x)
        
        # Return the result
        # This replaces result.to_dense()
        return output

if __name__ == "__main__":
    setup()
    
    # Input generation
    x = torch.randn(10, 10)
    
    model = TestModel()
    
    # Test Eager
    eager_output = model(x)
    print("Eager output:", eager_output)
    
    # Test Compiled
    # Replacing torch.compile(model)(x) logic
    compiled_model = torch.compile(model)
    compiled_output = compiled_model(x)
    print("Compiled output:", compiled_output)
    
    # Assertion to verify behavior
    assert torch.allclose(eager_output, compiled_output), "Eager and Compiled outputs differ"
    
    dist.destroy_process_group()