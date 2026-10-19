```python
import tensorflow as tf
import numpy as np
# Conversion: torch.distributed._tensor.init_device_mesh -> tensorflow.python.compiler.xla.experimental.xla_sharding.mesh_split
# Note: mesh_split is used to distribute the tensor across the mesh.
from tensorflow.python.compiler.xla.experimental.xla_sharding import mesh_split

# Conversion: torch.distributed.init_process_group -> tf.distribute.MultiWorkerMirroredStrategy
# Note: Standard TF distributed setup replaces explicit process group init.
strategy = tf.distribute.MultiWorkerMirroredStrategy()

if __name__ == '__main__':
    # dist.init_process_group(backend="nccl")
    # Handled by strategy initialization

    # rank = dist.get_rank()
    rank = strategy.cluster_resolver.task_id

    # torch.cuda.set_device(rank)
    # Device assignment is handled implicitly by the strategy scope

    with strategy.scope():
        # tensor = torch.arange(12).reshape(-1, 4).float().cuda()
        tensor = tf.cast(tf.reshape(tf.range(12), (-1, 4)), tf.float32)
        # print("Truth=", tensor.mean())
        print("Truth=", tf.reduce_mean(tensor))

        # mesh = init_device_mesh('cuda', (2,))
        # Define the device mesh topology (1D mesh of size 2)
        device_mesh = np.array([0, 1])

        # dt = distribute_tensor(tensor, device_mesh=mesh, placements=[Shard(0)])
        # Conversion: distribute_tensor with Shard(0) -> mesh_split with mapping [0]
        # Shard(0) implies splitting the 0th tensor dimension along the 0th mesh axis.
        dt = mesh_split(tensor, device_mesh, [0])

        # mean = dt.mean()
        mean = tf.reduce_mean(dt)

        # full = mean.full_tensor()
        # In XLA/SPMD, a scalar result from a reduction is replicated across devices,
        # so it is already "full" on the local device.
        full = mean

        print(rank, mean, full)
```