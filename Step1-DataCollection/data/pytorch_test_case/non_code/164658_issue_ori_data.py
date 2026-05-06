Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import torch
  File "/mnt/lvm/xuantengh/pytorch/torch/__init__.py", line 2078, in <module>
    from torch import amp as amp, random as random, serialization as serialization
  File "/mnt/lvm/xuantengh/pytorch/torch/random.py", line 22, in <module>
    def get_rng_state(device: torch.device = torch.device("cpu")) -> torch.Tensor:
                                             ~~~~~~~~~~~~^^^^^^^
  File "/mnt/lvm/xuantengh/pytorch/torch/fx/__init__.py", line 87, in <module>
    from torch.fx import immutable_collections
  File "/mnt/lvm/xuantengh/pytorch/torch/fx/immutable_collections.py", line 5, in <module>
    from torch.utils._pytree import (
    ...<8 lines>...
    )
  File "/mnt/lvm/xuantengh/pytorch/torch/utils/__init__.py", line 8, in <module>
    from torch.utils import (
    ...<5 lines>...
    )
  File "/mnt/lvm/xuantengh/pytorch/torch/utils/data/__init__.py", line 1, in <module>
    from torch.utils.data.dataloader import (
    ...<5 lines>...
    )
  File "/mnt/lvm/xuantengh/pytorch/torch/utils/data/dataloader.py", line 24, in <module>
    import torch.distributed as dist
  File "/mnt/lvm/xuantengh/pytorch/torch/distributed/__init__.py", line 131, in <module>
    from .device_mesh import DeviceMesh, init_device_mesh
  File "/mnt/lvm/xuantengh/pytorch/torch/distributed/device_mesh.py", line 70, in <module>
    torch.serialization.add_safe_globals([_MeshLayout])
    ^^^^^^^^^^^^^^^^^^^
AttributeError: partially initialized module 'torch' from '/mnt/lvm/xuantengh/pytorch/torch/__init__.py' has no attribute 'serialization' (most likely due to a circular import)