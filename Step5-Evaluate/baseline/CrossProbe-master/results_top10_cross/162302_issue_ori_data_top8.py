```python
import tensorflow as tf

# Conversion: torch.tensor -> tf.constant
A = tf.constant([
    [4.0, -3.0,  2.0, -1.0],
    [-1.0,  2.0, -3.0,  4.0],
    [3.0, -4.0,  1.0, -2.0],
    [-2.0,  1.0, -4.0,  3.0]
], dtype=tf.float32)

# Conversion: torch.linalg.tensorinv(A, ind=1) -> tf.linalg.inv(A)
# Note: ind=1 implies standard matrix inversion. TensorFlow does not have a direct tensorinv,
# but tf.linalg.inv is the mathematical equivalent for ind=1.
cpu_result = tf.linalg.inv(A)

# Conversion: A.to("cuda") -> tf.device('/GPU:0') context manager
with tf.device('/GPU:0'):
    gpu_result = tf.linalg.inv(A)

# Conversion: PyTorch prints tensor values directly. In TF eager mode, print(tensor) shows metadata.
# Using .numpy() to display values, similar to PyTorch's default print behavior.
print("CPU:\n", cpu_result.numpy())
print("GPU:\n", gpu_result.numpy())
# Conversion: gpu_result.cpu() -> implicit conversion to numpy handles device transfer
print("CPU / GPU elementwise ratio:\n", (cpu_result / gpu_result).numpy())
```