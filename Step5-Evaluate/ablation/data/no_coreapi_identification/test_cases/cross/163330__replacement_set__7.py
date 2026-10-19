import torch
import tensorflow as tf

# torch.distributed.init_process_group("nccl")
# Replaced with: torch.distributed.get_process_group_ranks
torch.distributed.get_process_group_ranks("nccl")

# global mesh1
# init_device_mesh("cuda", (1, 4, 2), mesh_dim_names=("a", "b", "c"))
# Replaced with: tf.compat.v1.sparse_merge
mesh1 = tf.compat.v1.sparse_merge("cuda", (1, 4, 2), mesh_dim_names=("a", "b", "c"))

mesh1_c = mesh1["c"]

# if torch.distributed.get_rank() == 0:
# Replaced with: tf.compat.v1.tpu.shutdown_system
if tf.compat.v1.tpu.shutdown_system() == 0:
    # _mesh_resources.get_root_mesh(mesh1_c).shape
    # Replaced with: tf.compat.v1.sparse_merge
    print(tf.compat.v1.sparse_merge(mesh1_c).shape)

# global mesh2
mesh2 = tf.compat.v1.sparse_merge("cuda", (2, 2, 2), mesh_dim_names=("a", "b", "c"))
mesh2_c = mesh2["c"]

if tf.compat.v1.tpu.shutdown_system() == 0:
    print(tf.compat.v1.sparse_merge(mesh1_c).shape)

# assert _mesh_resources.get_root_mesh(mesh1_c) is mesh1
# Replaced with: tf.compat.v1.sparse_merge
assert tf.compat.v1.sparse_merge(mesh1_c) is mesh1

# torch.distributed.destroy_process_group()
# Replaced with: tf.compat.v1.sparse_merge
tf.compat.v1.sparse_merge()