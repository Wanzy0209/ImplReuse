```python
import tensorflow as tf

# Conversion: torch.tensor -> tf.constant
input_tensor = tf.constant([[1, 2, 3], [3, 2, 1], [1, 1, 1]], dtype=tf.float32)

# Conversion: torch.pinverse -> tf.linalg.pinv
# Explicitly placing on CPU to match source logic
with tf.device('/CPU:0'):
    pinverse_tensor_cpu = tf.linalg.pinv(input_tensor)

print("input_tensor: ")
print(input_tensor)
print("CPU Pseudo-inverse:")
print(pinverse_tensor_cpu)

# Conversion: input_tensor.to('cuda') -> tf.device('/GPU:0') context
# Note: Requires GPU-enabled TensorFlow
with tf.device('/GPU:0'):
    # Mimic the move to GPU by creating an identity op on the GPU device
    input_tensor_gpu = tf.identity(input_tensor)
    pinverse_tensor_gpu = tf.linalg.pinv(input_tensor_gpu)

print("input_tensor_gpu: ")
print(input_tensor_gpu)
print("\nGPU Pseudo-inverse:")
print(pinverse_tensor_gpu)
```