import torch
from torch.distributed.tensor import empty
from torch.distributed._tensor.placement_types import Shard

def test_rng_semantics():
    device_mesh = torch.distributed.init_process_group(backend='nccl')._mesh
    torch.manual_seed(42)
    rng = torch.Generator(device='cuda').manual_seed(42)
    t1 = empty((2, 3), device_mesh=device_mesh, placements=[Shard(0)])
    t2 = empty((2, 3), device_mesh=device_mesh, placements=[Shard(0)])
    
    for i in range(2):
        torch.nn.init.uniform_(t1, 0.0, 1.0)
        torch.nn.init.uniform_(t2, 0.0, 1.0, generator=rng)
        assert t1.full_tensor().equal(t2.full_tensor())
    
    rng.manual_seed(55)
    torch.nn.init.uniform_(t1, 0.0, 1.0)
    torch.nn.init.uniform_(t2, 0.0, 1.0, generator=rng)
    assert t1.full_tensor().equal(t2.full_tensor())
    
    print('RNG state:', rng.get_state())
    print('Global state:', torch.cuda.get_rng_state())