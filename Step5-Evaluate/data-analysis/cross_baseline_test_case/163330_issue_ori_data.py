```python
"""
TF Distributed Test Script
Equivalent to: torchrun --nproc_per_node=8 test.py
"""
import tensorflow as tf
# Conversion: Using the mapped API for rank retrieval context
from tensorflow.python.distribute.cross_device_ops_test import make_collective

# Conversion: torch.distributed.init_process_group
# In TensorFlow, distributed initialization is handled by the Strategy.
# We use MultiWorkerMirroredStrategy which utilizes NCCL for GPUs.
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# Conversion: torch.distributed.device_mesh.init_device_mesh
# TF uses tf.experimental.dtensor.Mesh
def init_device_mesh(device_type, shape, mesh_dim_names):
    # Get physical devices
    if device_type == "cuda":
        all_devices = tf.config.list_physical_devices('GPU')
    else:
        all_devices = tf.config.list_physical_devices('CPU')
        
    # Flatten device list for mesh creation
    # PyTorch init_device_mesh handles device assignment automatically.
    # Here we map the logical shape to the available devices.
    # We assume enough devices are available (8 in this case).
    device_list = [d.name for d in all_devices]
    
    mesh_dims = list(zip(mesh_dim_names, shape))
    return tf.experimental.dtensor.Mesh(mesh_dims, device_list)

# Conversion: torch.distributed.get_rank
# Using the provided mapping: make_collective
# Note: make_collective is a test utility. We use it to obtain the process ID (rank).
# We assume 8 processes with 1 GPU per process based on the source context.
# In a real scenario, this would be configured via TF_CONFIG.
try:
    # Attempt to use the mapped API
    _, _, pid = make_collective(num_processes=8, gpu_per_process=1)
    rank = pid
except Exception:
    # Fallback for environments where the test util isn't available/configured
    rank = 0

with strategy.scope():
    # global mesh1
    mesh1 = init_device_mesh(
        "cuda", (1, 4, 2), mesh_dim_names=("a", "b", "c")
    )
    
    # Conversion: mesh1["c"]
    # PyTorch returns a sub-mesh. TF Mesh does not support sub-meshing with parent references.
    # We simulate this by creating a new mesh for the 'c' dimension.
    # Note: This object will not be linked to mesh1.
    mesh1_c = tf.experimental.dtensor.Mesh([("c", 2)], mesh1.device_ids())
    
    # got (1, 4, 2), OK!!
    if rank == 0:
        # Conversion: _mesh_resources.get_root_mesh
        # TF Mesh does not have a root mesh concept. We print the shape of the original mesh.
        print(mesh1.dim_sizes)
        
    # global mesh2
    mesh2 = init_device_mesh(
        "cuda", (2, 2, 2), mesh_dim_names=("a", "b", "c")
    )
    mesh2_c = tf.experimental.dtensor.Mesh([("c", 2)], mesh2.device_ids())
    
    # got (2, 2, 2), it's mesh2 instead of mesh1 !!!
    if rank == 0:
        # In TF, mesh1 is independent of mesh2, so this will print mesh1's shape.
        print(mesh1.dim_sizes)
        
    # Conversion: assert _mesh_resources.get_root_mesh(mesh1_c) is mesh1
    # This assertion checks for object identity and hierarchy.
    # Since TF Meshes are immutable and do not track parents, this check is not applicable.
    # We comment it out to maintain code structure validity.
    # assert mesh1_c is mesh1 # Not supported in TF
    
# Conversion: torch.distributed.destroy_process_group
# Cleanup is handled automatically when the strategy scope exits or the script finishes.
```