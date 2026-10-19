```python
import tensorflow as tf
from tensorflow.experimental import dtensor

if __name__ == '__main__':
    # dist.init_process_group(backend="nccl")
    # TensorFlow DTensor handles backend initialization automatically.
    
    # rank = dist.get_rank()
    # TF DTensor abstracts explicit rank access; logic is handled via Mesh.
    
    # torch.cuda.set_device(rank)
    # Device placement is managed by the Mesh configuration in TF.

    # tensor = torch.arange(12).reshape(-1, 4).float().cuda()
    tensor = tf.range(12, dtype=tf.float32)
    tensor = tf.reshape(tensor, (-1, 4))

    # print("Truth=", tensor.mean())
    print("Truth=", tf.reduce_mean(tensor).numpy())

    # mesh = init_device_mesh('cuda', (2,))
    # Create a device mesh with one dimension named 'x' of size 2 on GPU
    mesh = dtensor.create_mesh(['x'], [2], device_type='gpu')

    # dt = distribute_tensor(tensor, device_mesh=mesh, placements=[Shard(0)])
    # Define layout: Shard dimension 0 along mesh axis 'x'
    layout = dtensor.Layout([dtensor.Sharding('x')], rank=2)
    dt = dtensor.copy_to_mesh(tensor, layout, mesh)

    # mean = dt.mean()
    # Calculate mean. TF DTensor automatically performs all-reduce for global reduction.
    mean = tf.reduce_mean(dt)

    # full = mean.full_tensor()
    # Convert DTensor back to a local Tensor
    full = dtensor.unpack(mean)

    # print(rank, mean, full)
    # Rank is implicit in TF DTensor execution; printing the computed values.
    print("Result:", mean.numpy(), full.numpy())
```