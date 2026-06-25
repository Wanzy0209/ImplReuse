import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def worker(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:29500",
        world_size=world_size,
        rank=rank
    )
    dist.barrier()
    dist.destroy_process_group()
    torch.cuda.empty_cache()
    time.sleep(10)

def test():
    world_size = torch.cuda.device_count()
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    test()