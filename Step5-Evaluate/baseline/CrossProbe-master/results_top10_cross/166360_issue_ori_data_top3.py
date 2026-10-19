```python
import tensorflow as tf

# Conversion: torch.rand with device argument -> tf.random.uniform inside tf.device context
with tf.device('/device:GPU:0'):
    tensor1 = tf.random.uniform((2,), 0, 1)

with tf.device('/device:GPU:1'):
    tensor2 = tf.random.uniform((2,), 0, 1)

print(f"tensor1: {tensor1}, tensor2: {tensor2}")

# Conversion: comm.gather with destination -> tf.concat inside tf.device context
# Note: tf.concat will implicitly copy tensors to the destination device
with tf.device('/device:GPU:0'):
    combined_cuda = tf.concat([tensor1, tensor2], axis=0)
print(f'combined tensor - Cuda: {combined_cuda}')

with tf.device('/device:CPU:0'):
    combined_cpu = tf.concat([tensor1, tensor2], axis=0)
print(f'combined tensor - CPU: {combined_cpu}')
```