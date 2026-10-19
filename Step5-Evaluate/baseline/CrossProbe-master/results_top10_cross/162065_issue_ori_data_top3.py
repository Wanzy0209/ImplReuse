```python
import tensorflow as tf

# Create input tensor on CPU (default behavior)
# torch.tensor maps to tf.constant
input_tensor = tf.constant([[1, 2, 3], [3, 2, 1], [1, 1, 1]], dtype=tf.float32)

# Compute pseudo-inverse on CPU
# torch.pinverse maps to tf.linalg.pinv
pinverse_tensor_cpu = tf.linalg.pinv(input_tensor)

print("input_tensor: ")
print(input_tensor)
print("CPU Pseudo-inverse:")
print(pinverse_tensor_cpu)

# Move to GPU
# PyTorch's .to('cuda') is handled in TF by placing operations/tensors inside a device context
with tf.device('/GPU:0'):
    # Re-create tensor on GPU to simulate moving the data
    input_tensor_gpu = tf.constant([[1, 2, 3], [3, 2, 1], [1, 1, 1]], dtype=tf.float32)

    # Compute pseudo-inverse on GPU
    pinverse_tensor_gpu = tf.linalg.pinv(input_tensor_gpu)

print("input_tensor_gpu: ")
print(input_tensor_gpu)
print("\nGPU Pseudo-inverse:")
print(pinverse_tensor_gpu)
```