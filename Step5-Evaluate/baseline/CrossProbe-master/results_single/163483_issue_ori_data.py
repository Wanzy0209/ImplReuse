# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch

torch.distributed.init_process_group(backend='nccl')
rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
torch.cuda.set_device(rank)

x = torch.arange(0, 16).reshape(2, 2, 2, 2).cuda().to(memory_format=torch.channels_last)
x_list = [torch.zeros_like(x) for _ in range(world_size)]
torch.distributed.all_gather(x_list, x)

print('rank_{}: {}\n x:\n{}\n gathered_x:\n{}\n'.format(rank, torch.equal(x, x_list[rank]), x.storage(), x_list[rank].storage()))