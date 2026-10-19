```python
"""
tensorflow test.py
"""
import tensorflow as tf

# Initialize distributed strategy (equivalent to init_process_group)
# Using MirroredStrategy for local multi-GPU (nccl equivalent)
strategy = tf.distribute.MirroredStrategy()

# Mocking the PyTorch DeviceMesh behavior for TensorFlow
# as TF DTensor does not have the exact same sub-mesh/parent hierarchy.
class _MeshResources:
    @staticmethod
    def get_root_mesh(mesh):
        # In our wrapper, we store the parent reference
        return mesh._parent if mesh._parent else mesh

_mesh_resources = _MeshResources()

def init_device_mesh(device_type, mesh_shape, mesh_dim_names):
    # Wrapper to mimic PyTorch's DeviceMesh
    class DeviceMesh:
        def __init__(self, shape, dim_names, parent=None):
            self.shape = shape
            self.dim_names = dim_names
            self._parent = parent
        
        def __getitem__(self, name):
            # Mimic sub-mesh creation which retains reference to root
            idx = self.dim_names.index(name)
            sub_shape = (self.shape[idx],)
            return DeviceMesh(sub_shape, (name,), parent=self)
    
    return DeviceMesh(mesh_shape, mesh_dim_names)

# Run the distributed logic
# Note: In TF, we typically use strategy.scope() to replicate variables.
# The logic here is control flow, so we run it once per worker/replica.
# However, to mimic torchrun (one script per process), we rely on strategy's execution.

# To strictly follow the structure, we execute the logic.
# Since this is a script, we can just run it. The strategy handles the distribution.

# global mesh1
mesh1 = init_device_mesh(
    "cuda", (1, 4, 2), mesh_dim_names=("a", "b", "c")
)
mesh1_c = mesh1["c"]

# got (1, 4, 2), OK!!
# Check for rank 0 (chief replica)
# Note: get_replica_context() must be called inside strategy.scope() or a tf.function
with strategy.scope():
    # We need to use tf.distribute.get_replica_context() carefully.
    # For a simple script translation, we can check the local device ID or similar.
    # But standard way is:
    
    def step_fn():
        # global mesh1 (re-defined or accessed from outer scope if not tf.Variable)
        # Since mesh1 is a python object, it's available.
        
        # tf.print is preferred in distributed, but print works for rank 0 usually.
        if tf.distribute.get_replica_context().replica_id_in_sync_group == 0:
            print(_mesh_resources.get_root_mesh(mesh1_c).shape)
        
        # global mesh2
        mesh2 = init_device_mesh(
            "cuda", (2, 2, 2), mesh_dim_names=("a", "b", "c")
        )
        mesh2_c = mesh2["c"]
        
        # got (2, 2, 2), it's mesh2 instead of mesh1 !!!
        if tf.distribute.get_replica_context().replica_id_in_sync_group == 0:
            print(_mesh_resources.get_root_mesh(mesh1_c).shape)
        
        # raise error here
        assert _mesh_resources.get_root_mesh(mesh1_c) is mesh1

    strategy.run(step_fn)

# No explicit destroy_process_group needed
```