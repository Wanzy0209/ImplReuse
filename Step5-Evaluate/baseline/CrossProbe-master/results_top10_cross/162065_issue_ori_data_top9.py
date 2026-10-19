```python
import tensorflow as tf

# PyTorch: torch.tensor(...) -> TensorFlow: tf.constant(...)
input_tensor = tf.constant([[1, 2, 3], [3, 2, 1], [1, 1, 1]], dtype=tf.float32)

# PyTorch: torch.pinverse -> TensorFlow: tf.linalg.pinv
# Explicitly placing on CPU to match PyTorch default behavior
with tf.device('/CPU:0'):
    pinverse_tensor_cpu = tf.linalg.pinv(input_tensor)

print("input_tensor: ")
print(input_tensor)
print("CPU Pseudo-inverse:")
print(pinverse_tensor_cpu)

# PyTorch: input_tensor.to('cuda') -> TensorFlow: tf.device('/GPU:0') context
# Note: TensorFlow handles device placement via context managers. 
# We use tf.identity to ensure the tensor is copied to the GPU within the context.
with tf.device('/GPU:0'):
    input_tensor_gpu = tf.identity(input_tensor)
    pinverse_tensor_gpu = tf.linalg.pinv(input_tensor_gpu)

print("input_tensor_gpu: ")
print(input_tensor_gpu)
print("\nGPU Pseudo-inverse:")
print(pinverse_tensor_gpu)
```