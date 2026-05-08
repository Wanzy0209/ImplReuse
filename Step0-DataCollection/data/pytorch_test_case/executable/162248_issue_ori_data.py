import torch
print(torch.__version__)
import os
import multiprocessing as mp
import torch.distributed as dist

world_size                      = 2
os.environ['MASTER_ADDR']       = '127.0.0.1'
os.environ['MASTER_PORT']       = '27015'
os.environ['WORLD_SIZE']        = str(world_size)

data = [local_rank for local_rank in range(world_size)]

def run(local_rank):
        group   = dist.init_process_group(backend='gloo', rank=local_rank)
        output  = list(torch.empty([world_size], dtype=torch.int64).chunk(world_size))
        input   = list((torch.arange(world_size) + local_rank * world_size).chunk(world_size))
        dist.all_to_all(output, input, group=group)
        return output

with mp.Pool(world_size) as p:
        for x in p.map(run, list(range(world_size)), 1):
                print(x)