```python
import tensorflow as tf

# Conversion: torch.rand(2, device=torch.device("cuda:0"))
# Note: In TensorFlow, device placement is handled via context managers.
with tf.device('/device:GPU:0'):
    tensor1 = tf.random.uniform((2,), minval=0, maxval=1)

# Conversion: torch.rand(2, device=torch.device("cuda:1"))
with tf.device('/device:GPU:1'):
    tensor2 = tf.random.uniform((2,), minval=0, maxval=1)

print(f"tensor1: {tensor1}, tensor2: {tensor2}")

# Conversion: comm.gather([tensor1, tensor2], destination = torch.device("cuda:0"))
# Note: PyTorch's gather concatenates tensors along a dimension (default 0) onto the destination device.
# We simulate this by placing the concatenation operation on the target device.
with tf.device('/device:GPU:0'):
    combined_tensor_cuda = tf.concat([tensor1, tensor2], axis=0)

print(f'combined tensor - Cuda: {combined_tensor_cuda}')

# Conversion: comm.gather([tensor1, tensor2], destination = torch.device("cpu"))
with tf.device('/CPU:0'):
    combined_tensor_cpu = tf.concat([tensor1, tensor2], axis=0)

print(f'combined tensor - CPU: {combined_tensor_cpu}')
```