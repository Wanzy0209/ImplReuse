```python
import tensorflow as tf

# Conversion: torch.rand with device placement
# PyTorch uses device argument, TF uses context manager for device placement
with tf.device('/GPU:0'):
    tensor1 = tf.random.uniform((2,))
with tf.device('/GPU:1'):
    tensor2 = tf.random.uniform((2,))

print(f"tensor1: {tensor1}, tensor2: {tensor2}")

# Conversion: comm.gather to destination device
# In TF, placing the concat operation on the destination device gathers the tensors there.
with tf.device('/GPU:0'):
    combined_cuda = tf.concat([tensor1, tensor2], axis=0)
print(f'combined tensor - Cuda: {combined_cuda}')

with tf.device('/CPU:0'):
    combined_cpu = tf.concat([tensor1, tensor2], axis=0)
print(f'combined tensor - CPU: {combined_cpu}')
```