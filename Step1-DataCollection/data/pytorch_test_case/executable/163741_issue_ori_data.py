import torch
import os
import glob
import time
import torch.distributed as dist
import torch.multiprocessing as mp

def worker(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl", 
        init_method="tcp://127.0.0.1:29500",
        world_size=world_size,
        rank=rank
    )
    dist.barrier()

    device = f"cuda:{rank}"
    # do something
    print("to sleep")
    time.sleep(5)
    
    dist.destroy_process_group()

    # sleep for observing nvidia-smi
    time.sleep(100)

def test_nccl():
    world_size = torch.cuda.device_count()
    mp.spawn(
        worker,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )

if __name__ == '__main__':
    test_nccl()