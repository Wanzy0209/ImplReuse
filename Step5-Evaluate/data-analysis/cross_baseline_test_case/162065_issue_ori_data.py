```python
import tensorflow as tf

# Conversion: torch.tensor -> tf.constant
input_tensor = tf.constant([[1, 2, 3], [3, 2, 1], [1, 1, 1]], dtype=tf.float32)

# Conversion: torch.pinverse -> tf.linalg.pinv
pinverse_tensor_cpu = tf.linalg.pinv(input_tensor)

print("input_tensor: ")
# Conversion: Use .numpy() to print values like PyTorch
print(input_tensor.numpy())
print("CPU Pseudo-inverse:")
print(pinverse_tensor_cpu.numpy())

# Conversion: input_tensor.to('cuda') -> tf.device context manager
# Note: This block requires a GPU to be available.
with tf.device('/GPU:0'):
    # tf.identity creates a copy of the tensor on the specified device
    input_tensor_gpu = tf.identity(input_tensor)
    pinverse_tensor_gpu = tf.linalg.pinv(input_tensor_gpu)

print("input_tensor_gpu: ")
print(input_tensor_gpu.numpy())
print("\nGPU Pseudo-inverse:")
print(pinverse_tensor_gpu.numpy())
```