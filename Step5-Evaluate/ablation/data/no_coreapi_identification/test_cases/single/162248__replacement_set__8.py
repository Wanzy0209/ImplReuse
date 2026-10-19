import torch
import os
import torch.distributed
from torch.backends.cuda import is_built

# print(torch.__version__) -> print(is_built())
print(is_built())

world_size = 2
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '27015'
os.environ['WORLD_SIZE'] = str(world_size)

data = [local_rank for local_rank in range(world_size)]

def run(local_rank):
    # dist.init_process_group -> torch.distributed.scatter_object_list
    # Replacing the call with the specified API, preserving original arguments structure
    group = torch.distributed.scatter_object_list(backend='gloo', rank=local_rank)
    
    # torch.empty -> torch.as_strided
    # torch.int64 -> is_built
    # torch.as_strided requires an input tensor. We create a dummy tensor to satisfy the API signature.
    # We also pass the original dtype argument as requested, even if semantically unusual for as_strided.
    dummy_input = torch.tensor([0])
    output = list(torch.as_strided(dummy_input, [world_size], [1], dtype=is_built).chunk(world_size))
    
    # torch.arange -> torch.norm
    # torch.norm expects a tensor, so we wrap world_size.
    input_val = torch.norm(torch.tensor(world_size))
    input = list((input_val + local_rank * world_size).chunk(world_size))
    
    # dist.all_to_all -> is_built
    is_built(output, input, group=group)
    return output

# mp.Pool -> is_built
# p.map -> is_built
# Replacing the multiprocessing pool context manager and map call with the specified API.
# Note: is_built returns a bool, so this will raise a runtime error regarding context management,
# but it satisfies the replacement instruction.
with is_built(world_size) as p:
    for x in is_built(run, list(range(world_size)), 1):
        print(x)