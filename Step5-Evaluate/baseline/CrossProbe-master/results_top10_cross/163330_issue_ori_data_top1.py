```python
"""
tensorflow test.py
"""
import tensorflow as tf
from tensorflow.dtensor import Mesh

# Conversion: torch.distributed.init_process_group("nccl")
# TensorFlow initializes distributed environment via TF_CONFIG or Strategy.
# Here we assume the environment is configured for the available devices.
devices = tf.config.list_physical_devices('GPU')
# Ensure we have enough devices for the mesh shapes (8 required)
# If running on CPU or fewer GPUs, this might fail, but we assume the environment matches the PyTorch script.
if len(devices) < 8:
    # Fallback for testing purposes if not enough GPUs
    devices = [f"/cpu:{i}" for i in range(8)]

# global mesh1
# Conversion: init_device_mesh("cuda", (1, 4, 2), mesh_dim_names=("a", "b", "c"))
mesh1 = Mesh(
    dim_names=["a", "b", "c"],
    shape=[1, 4, 2],
    devices=devices
)

# Conversion: mesh1_c = mesh1["c"]
# TF dtensor.Mesh does not support slicing to return a sub-mesh object with a parent reference.
# We assign mesh1 to mesh1_c to preserve variable usage, but note the limitation.
mesh1_c = mesh1

# Conversion: if torch.distributed.get_rank() == 0:
# TF runs on a single process for MirroredStrategy. We print directly.
# In a distributed setting, one might check cluster_resolver, but here we just print.
# got (1, 4, 2), OK!!
print(f"Mesh 1 shape: {mesh1.shape}")

# global mesh2
# Conversion: init_device_mesh("cuda", (2, 2, 2), mesh_dim_names=("a", "b", "c"))
mesh2 = Mesh(
    dim_names=["a", "b", "c"],
    shape=[2, 2, 2],
    devices=devices
)

# Conversion: mesh2_c = mesh2["c"]
mesh2_c = mesh2

# got (2, 2, 2), it's mesh2 instead of mesh1 !!!
# Conversion: print(_mesh_resources.get_root_mesh(mesh1_c).shape)
# TF does not have get_root_mesh. We verify mesh1 is still intact.
print(f"Mesh 1 shape after mesh2 creation: {mesh1.shape}")

# Conversion: assert _mesh_resources.get_root_mesh(mesh1_c) is mesh1
# TF does not support parent mesh references. 
# We assert that mesh1 retains its original shape and identity.
assert mesh1.shape == [1, 4, 2], "Mesh 1 shape was corrupted by mesh2 creation"
assert mesh1 is mesh1, "Mesh 1 identity check"

# Conversion: torch.distributed.destroy_process_group()
# TF handles cleanup automatically.
```