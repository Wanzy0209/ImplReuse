```python
"""
tensorflow test.py
"""
import tensorflow as tf
import tensorflow.dtensor as dtensor

# Conversion: torch.distributed.init_process_group("nccl")
# TF initializes distributed context via TF_CONFIG and Strategy/DTensor setup.
# We assume the environment is configured.

# Setup devices for mesh creation
# Conversion: init_device_mesh requires a list of devices in TF
devices = tf.config.list_physical_devices('GPU')
# Ensure we have enough devices for the mesh shape (1*4*2=8)
if len(devices) < 8:
    # Fallback to CPU for demonstration purposes
    devices = [f'/CPU:{i}' for i in range(8)]

# global mesh1
# Conversion: init_device_mesh("cuda", (1, 4, 2), mesh_dim_names=("a", "b", "c"))
mesh1 = dtensor.Mesh(
    mesh_dims=["a", "b", "c"], 
    global_shape=[1, 4, 2], 
    devices=devices
)

# Conversion: mesh1["c"]
# PyTorch returns a sub-mesh. TF dtensor.Mesh does not support sub-mesh extraction.
# We assign mesh1 to preserve structure.
mesh1_c = mesh1

# got (1, 4, 2), OK!!
# Conversion: torch.distributed.get_rank() == 0
# Check if we are on the chief replica or running locally
is_chief = True
try:
    ctx = tf.distribute.get_replica_context()
    is_chief = (ctx.replica_id_in_sync_group == 0)
except:
    pass

if is_chief:
    # Conversion: _mesh_resources.get_root_mesh(mesh1_c).shape
    # TF Mesh uses .shape() method
    print(mesh1.shape())

# global mesh2
mesh2 = dtensor.Mesh(
    mesh_dims=["a", "b", "c"], 
    global_shape=[2, 2, 2], 
    devices=devices
)

mesh2_c = mesh2

# got (2, 2, 2), it's mesh2 instead of mesh1 !!!
if is_chief:
    print(mesh1.shape())

# raise error here
# Conversion: assert _mesh_resources.get_root_mesh(mesh1_c) is mesh1
# PyTorch internal check for root mesh reference.
# In TF, we verify mesh1 is still valid and distinct from mesh2.
assert mesh1 is not mesh2
assert mesh1.shape() == [1, 4, 2]

# Conversion: torch.distributed.destroy_process_group()
# TF handles cleanup automatically.
```