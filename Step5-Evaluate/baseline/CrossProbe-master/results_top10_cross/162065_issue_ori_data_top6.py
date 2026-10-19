```python
import tensorflow as tf

# torch.tensor -> tf.constant
input_tensor = tf.constant([[1, 2, 3], [3, 2, 1], [1, 1, 1]], dtype=tf.float32)

# torch.pinverse -> tf.linalg.pinv
pinverse_tensor_cpu = tf.linalg.pinv(input_tensor)

print("input_tensor: ")
print(input_tensor)
print("CPU Pseudo-inverse:")
print(pinverse_tensor_cpu)

# input_tensor.to('cuda') -> tf.device context manager
# In TensorFlow, device placement is typically handled via context managers rather than methods on the tensor.
with tf.device('/GPU:0'):
    # tf.identity is used to explicitly place a copy of the tensor on the GPU
    input_tensor_gpu = tf.identity(input_tensor)

    # torch.pinverse -> tf.linalg.pinv
    pinverse_tensor_gpu = tf.linalg.pinv(input_tensor_gpu)

print("input_tensor_gpu: ")
print(input_tensor_gpu)
print("\nGPU Pseudo-inverse:")
print(pinverse_tensor_gpu)
```