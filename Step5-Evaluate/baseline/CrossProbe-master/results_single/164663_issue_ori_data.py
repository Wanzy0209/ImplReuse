# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import os
import torch
import torch.distributed as dist
import torch.nn as nn

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.tensor.experimental import implicit_replication
from torch.distributed._tools.fsdp2_mem_tracker import FSDPMemTracker


class TestModule(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.output(x)
        return x


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    dist.init_process_group(backend="nccl")

    d_model = 128

    model = TestModule(d_model)
    model = model.to('cuda:0')
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))

    fully_shard([model.norm, model.output], mesh=mesh)   # shards the RMSNorm's parameters/state across ranks
    fully_shard(model, mesh=mesh)  # shards the Linear's parameters/state across ranks
    print(model)

    tracker = FSDPMemTracker(model)

    with tracker, implicit_replication():
        x = torch.randn(16, d_model, device='cuda:0')
        y = model(x)
        loss = y.sum()
        loss.backward()


if __name__ == "__main__":
    main()