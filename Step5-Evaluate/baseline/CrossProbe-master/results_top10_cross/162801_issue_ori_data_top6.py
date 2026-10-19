```python
import tensorflow as tf

print(tf.__version__, flush=True)

shape = (3,)
scale = 0.08547895223232808
zero_point = 70
data_fp = tf.constant([-5.98352670669555664062500000000000e+00, -5.72708988189697265625000000000000e+00, -5.72708988189697265625000000000000e+00])

# Conversion: torch.quantize_per_tensor with scale/zero_point
# TF uses min/max. We derive min/max from scale/zero_point and the dtype range (0-3 for quint2x4).
min_val = (0 - zero_point) * scale
max_val = (3 - zero_point) * scale

# Use tf.quint8 as the closest standard dtype to quint2x4 (values 0-3 fit in quint8)
tensor_q1, _, _ = tf.quantization.quantize(data_fp, min_val, max_val, tf.quint8)

input = [[tensor_q1, tensor_q1], {}, [], {}]

# Conversion: torch.equal returns a single boolean.
# tf.equal returns a boolean tensor. We use reduce_all to get a single boolean.
tf.reduce_all(tf.equal(*input[0], **input[1]))
```