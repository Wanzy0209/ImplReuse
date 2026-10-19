```python
import tensorflow as tf
print(tf.__version__, flush=True)

shape = (3,)
scale = 0.08547895223232808
zero_point = 70
data_fp = tf.constant([-5.98352670669555664062500000000000e+00, -5.72708988189697265625000000000000e+00, -5.72708988189697265625000000000000e+00])

# Conversion: torch.quantize_per_tensor to tf.quantization.quantize
# PyTorch uses scale/zero_point, TF uses min/max. We convert scale/zp to min/max.
# PyTorch dtype quint2x4 (2-bit) is not natively supported in TF; using quint8 (8-bit) as substitute.
# Range for quint2x4 is 0 to 3.
min_range = (0 - zero_point) * scale
max_range = (3 - zero_point) * scale

tensor_q1, _, _ = tf.quantization.quantize(
    data_fp,
    min_range=min_range,
    max_range=max_range,
    T=tf.quint8
)

input = [[tensor_q1, tensor_q1], {}, [], {}]

# Conversion: torch.equal to tf.reduce_all(tf.equal(...))
# torch.equal returns a scalar bool, tf.equal returns a bool tensor.
# reduce_all aggregates the result to match the scalar behavior.
tf.reduce_all(tf.equal(*input[0], **input[1]))
```