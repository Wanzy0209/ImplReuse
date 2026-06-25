import torch
import torch.distributed as dist

def init_process(rank, world_size):
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(rank, torch.cuda.memory_allocated() / 1024**2, "MB before destroy")
    torch.cuda.synchronize()
    dist.destroy_process_group()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(init_process, args=(world_size,), nprocs=world_size)