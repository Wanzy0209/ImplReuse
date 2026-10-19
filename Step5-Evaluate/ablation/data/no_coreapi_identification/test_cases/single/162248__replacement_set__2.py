import torch
import os

# torch.__version__ -> torch.backends.mkldnn.is_available
print(torch.backends.mkldnn.is_available)

# torch.distributed -> torch.backends.mkldnn.is_available
from torch.backends.mkldnn import is_available as dist

world_size = 2
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '27015'
os.environ['WORLD_SIZE'] = str(world_size)

data = [local_rank for local_rank in range(world_size)]

def run(local_rank):
    # dist.init_process_group -> torch.hub.load_state_dict_from_url
    group = torch.hub.load_state_dict_from_url(backend='gloo', rank=local_rank)
    
    # torch.empty -> torch.isnan
    # torch.int64 -> torch.backends.mkldnn.is_available
    output = list(torch.isnan([world_size], dtype=torch.backends.mkldnn.is_available).chunk(world_size))
    
    # torch.arange -> torch.std
    input = list((torch.std(world_size) + local_rank * world_size).chunk(world_size))
    
    # dist.all_to_all -> torch.backends.mkldnn.is_available
    torch.backends.mkldnn.is_available(output, input, group=group)
    return output

# mp.Pool -> torch.backends.mkldnn.is_available
with torch.backends.mkldnn.is_available(world_size) as p:
    # p.map -> torch.backends.mkldnn.is_available
    for x in torch.backends.mkldnn.is_available(run, list(range(world_size)), 1):
        print(x)