```python
import tensorflow as tf

# Conversion: torch.rand -> tf.random.uniform
# Conversion: torch.device -> tf.device context manager
with tf.device('/device:GPU:0'):
    tensor1 = tf.random.uniform((2,))

with tf.device('/device:GPU:1'):
    tensor2 = tf.random.uniform((2,))

print(f"tensor1: {tensor1}, tensor2: {tensor2}")

# Conversion: comm.gather -> tf.concat
# Note: In TensorFlow, the 'destination' is handled by placing the operation
# (tf.concat) within the target device's scope. TF handles the data transfer.
with tf.device('/device:GPU:0'):
    combined_tensor_cuda = tf.concat([tensor1, tensor2], axis=0)

print(f'combined tensor - Cuda: {combined_tensor_cuda}')

with tf.device('/device:CPU:0'):
    combined_tensor_cpu = tf.concat([tensor1, tensor2], axis=0)

print(f'combined tensor - CPU: {combined_tensor_cpu}')
```