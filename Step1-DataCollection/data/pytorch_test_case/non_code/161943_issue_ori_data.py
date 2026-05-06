import torch
from torch import nn
from torch import distributed as dist
from torch.distributed.device_mesh import DeviceMesh


dist.init_process_group(backend='nccl', init_method='env://')
torch.cuda.set_device(dist.get_rank())

tensor = torch.randn(2)

mesh = DeviceMesh(
    device_type="cuda",
    mesh=[[0, 1, 2, 3], [4, 5, 6, 7]],
    mesh_dim_names=('fsdp', 'ep')
)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.mesh = mesh

    def forward(self, x):
        # This line causes error during JVP
        self.mesh['ep'].get_group()  
        return x

# Works in eager mode
mesh['ep'].get_group()  # Passes without error

# Fails with JVP
torch.func.jvp(
    Model(),
    (tensor,),
    (tensor,)
)  # Raises RuntimeError