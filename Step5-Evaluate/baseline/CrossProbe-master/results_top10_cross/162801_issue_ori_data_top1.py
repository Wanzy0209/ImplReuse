```python
import tensorflow as tf

print(tf.__version__, flush=True)

shape = (3,)
scale = 0.08547895223232808
zero_point = 70
data_fp = tf.constant([-5.98352670669555664062500000000000e+00, -5.72708988189697265625000000000000e+00, -5.72708988189697265625000000000000e+00])

# Conversion: PyTorch quint2x4 (4-bit) -> TensorFlow quint8 (8-bit) as 4-bit is not standard in TF core.
# Calculate min/max for TF quantization from scale/zero_point
min_range = (0 - zero_point) * scale
max_range = (255 - zero_point) * scale
tensor_q1, _, _ = tf.quantization.quantize(data_fp, min_range, max_range, tf.quint8)

input = [[tensor_q1, tensor_q1], {}, [], {}]

# torch.equal returns a boolean scalar. tf.equal returns a boolean tensor.
# Use reduce_all to simulate torch.equal behavior.
tf.reduce_all(tf.equal(*input[0]))
```