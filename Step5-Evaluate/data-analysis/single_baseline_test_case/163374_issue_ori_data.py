# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
from torch.distributed._tensor import Shard, distribute_tensor, init_device_mesh, DTensor, Replicate
import torch
from torch import distributed as dist
from torch.utils.debug_mode import DebugMode


if __name__ == '__main__':
    dist.init_process_group(backend="nccl", world_size=2)
    rank = dist.get_rank()
    mesh = init_device_mesh('cuda', (2,))

    tensor = torch.ones(12, 12, device="cuda")
    in_dtensor = distribute_tensor(tensor, mesh, [Shard(0)]) 

    partial_dt = in_dtensor.sum()
    with DebugMode(record_torchfunction=False) as debug_mode:
        out = partial_dt.clamp_(max=2)
        full = out.full_tensor()

    if rank == 0:
        print(debug_mode.debug_string())
        # redistribute_input(0, [P] -> [R])
        #   _c10d_functional::all_reduce(t: f32[], sum, 0)
        #   _c10d_functional::wait_tensor(t: f32[])
        # aten::clamp_(t: f32[], None, 2)
        # _c10d_functional::all_reduce(t: f32[], sum, 0)
        # _c10d_functional::wait_tensor(t: f32[])
        # aten::view(t: f32[], [])

        print(out.placements)
        # (Partial(sum),)
        # Expected (Replicate(),)

        print(full)
        # tensor(144., device='cuda:0')
        # Expected tensor(2., device='cuda:0')