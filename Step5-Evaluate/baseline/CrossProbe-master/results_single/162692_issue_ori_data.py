# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
from torch.distributed._tensor import Shard, distribute_tensor, init_device_mesh
import torch
from torch import distributed as dist

if __name__ == '__main__':
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    tensor = torch.arange(12).reshape(-1, 4).float().cuda()
    print("Truth=", tensor.mean())

    mesh = init_device_mesh('cuda', (2,))
    dt = distribute_tensor(tensor, device_mesh=mesh, placements=[Shard(0)])

    mean = dt.mean()
    full = mean.full_tensor()
    print(rank, mean, full)