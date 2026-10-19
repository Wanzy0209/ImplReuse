```python
import tensorflow as tf
import numpy as np

# TensorFlow does not require explicit process group initialization like PyTorch.
# It handles initialization automatically based on the environment variables and visible devices.
# dist.init_process_group(backend="nccl")

if __name__ == '__main__':
    # In TensorFlow, rank is context-dependent and accessed via ReplicaContext inside strategy.run.
    # It is not a global variable like in PyTorch's process-based model.
    # rank = dist.get_rank()
    
    # TensorFlow handles device assignment automatically based on the distribution strategy.
    # torch.cuda.set_device(rank)

    # Create the tensor
    # PyTorch: torch.arange(12).reshape(-1, 4).float().cuda()
    # Note: We create it on CPU first, then distribute to GPU mesh.
    tensor = tf.range(12, dtype=tf.float32)
    tensor = tf.reshape(tensor, (-1, 4))
    
    # Print the mean of the local tensor (Truth)
    # In PyTorch this is on CUDA, here it is on CPU by default.
    print("Truth=", tf.reduce_mean(tensor))

    # Initialize device mesh
    # PyTorch: mesh = init_device_mesh('cuda', (2,))
    # TF: create_mesh. We assume 2 GPUs are available.
    mesh = tf.experimental.dtensor.create_mesh(
        ['x'],  # Axis name
        [2],    # Mesh size
        device_type='GPU'
    )

    # Distribute tensor
    # PyTorch: dt = distribute_tensor(tensor, device_mesh=mesh, placements=[Shard(0)])
    # TF: copy_to_mesh with a Layout.
    # Shard(0) corresponds to Sharding the first dimension on the mesh axis 'x'.
    layout = tf.experimental.dtensor.Layout(
        [tf.experimental.dtensor.Sharded('x'), tf.experimental.dtensor.UNSHARDED],
        mesh
    )
    dt = tf.experimental.dtensor.copy_to_mesh(tensor, mesh, layout=layout)

    # Calculate mean
    # PyTorch: mean = dt.mean()
    mean = tf.reduce_mean(dt)

    # Gather full tensor
    # PyTorch: full = mean.full_tensor()
    # In TF DTensor, a scalar result from a reduction is fully replicated.
    # No explicit gather operation is needed.
    full = mean

    # Print results
    # Since 'rank' is not globally available in the same way, we print the values directly.
    print("Result:", mean, full)
```