# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
"""
torchrun --nproc_per_node=8 test.py
"""
import torch
from torch.distributed.device_mesh import _mesh_resources
from torch.distributed.device_mesh import init_device_mesh

torch.distributed.init_process_group("nccl")
# global mesh1
mesh1 = init_device_mesh(
    "cuda", (1, 4, 2), mesh_dim_names=("a", "b", "c")
)
mesh1_c = mesh1["c"]
# got (1, 4, 2), OK!!
if torch.distributed.get_rank() == 0:
    print(_mesh_resources.get_root_mesh(mesh1_c).shape)
# global mesh2
mesh2 = init_device_mesh(
    "cuda", (2, 2, 2), mesh_dim_names=("a", "b", "c")
)
mesh2_c = mesh2["c"]
# got (2, 2, 2), it's mesh2 instead of mesh1 !!!
if torch.distributed.get_rank() == 0:
    print(_mesh_resources.get_root_mesh(mesh1_c).shape)

# raise error here
assert _mesh_resources.get_root_mesh(mesh1_c) is mesh1
torch.distributed.destroy_process_group()