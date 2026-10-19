# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
from torch.utils._python_dispatch import TorchDispatchMode
import torch.distributed as dist
from torch.distributed._distributed_c10d import FakeProcessGroup
from torch.testing._internal.distributed.fake_pg import FakeStore


class SimpleTensorMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        print(func)
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


fake_pg = FakeProcessGroup(rank=0, world_size=3)
with SimpleTensorMode():
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    dist.all_reduce(tensor, group=fake_pg)