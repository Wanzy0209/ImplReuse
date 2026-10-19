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

# Conversion: input_tensor.to('cuda') -> tf.device context manager
# Note: In TensorFlow, device placement is typically handled via context managers
# rather than moving the tensor object itself.
with tf.device('/GPU:0'):
    input_tensor_gpu = input_tensor
    pinverse_tensor_gpu = tf.linalg.pinv(input_tensor_gpu)

print("input_tensor_gpu: ")
print(input_tensor_gpu)
print("\nGPU Pseudo-inverse:")
print(pinverse_tensor_gpu)
```