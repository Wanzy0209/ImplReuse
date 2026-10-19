```python
import tensorflow as tf

# Conversion: torch.rand -> tf.random.uniform
# Conversion: torch.device("cuda:0") -> tf.device("/gpu:0")
with tf.device("/gpu:0"):
    tensor1 = tf.random.uniform((2,))
with tf.device("/gpu:1"):
    tensor2 = tf.random.uniform((2,))

print(f"tensor1: {tensor1}, tensor2: {tensor2}")

# Conversion: comm.gather to destination -> copy tensors to destination and concat
# PyTorch's comm.gather concatenates tensors along a dimension (default 0) on the destination device.
with tf.device("/gpu:0"):
    # tensor1 is already on gpu:0, tensor2 needs to be copied
    tensor2_gpu0 = tf.identity(tensor2)
    combined_cuda = tf.concat([tensor1, tensor2_gpu0], axis=0)
print(f'combined tensor - Cuda: {combined_cuda}')

with tf.device("/cpu:0"):
    tensor1_cpu = tf.identity(tensor1)
    tensor2_cpu = tf.identity(tensor2)
    combined_cpu = tf.concat([tensor1_cpu, tensor2_cpu], axis=0)
print(f'combined tensor - CPU: {combined_cpu}')
```