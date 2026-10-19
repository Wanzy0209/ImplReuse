```python
import tensorflow as tf

# Conversion: torch.tensor -> tf.constant
input_tensor = tf.constant([[1, 2, 3], [3, 2, 1], [1, 1, 1]], dtype=tf.float32)

# Conversion: torch.pinverse -> tf.linalg.pinv
pinverse_tensor_cpu = tf.linalg.pinv(input_tensor)

print("input_tensor: ")
print(input_tensor)
print("CPU Pseudo-inverse:")
print(pinverse_tensor_cpu)

# Conversion: input_tensor.to('cuda') -> tf.device('/GPU:0') context
# In TensorFlow, device placement is handled via context managers or explicit placement.
# We wrap the GPU operations in a device block.
with tf.device('/GPU:0'):
    # Explicitly placing the tensor on GPU to mimic input_tensor_gpu
    input_tensor_gpu = tf.identity(input_tensor)
    pinverse_tensor_gpu = tf.linalg.pinv(input_tensor_gpu)

print("input_tensor_gpu: ")
print(input_tensor_gpu)
print("\nGPU Pseudo-inverse:")
print(pinverse_tensor_gpu)
```