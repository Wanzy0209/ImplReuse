```python
import tensorflow as tf
print(tf.__version__, flush=True)

shape = (3,)
scale = 0.08547895223232808
zero_point = 70
data_fp = tf.constant([-5.98352670669555664062500000000000e+00, -5.72708988189697265625000000000000e+00, -5.72708988189697265625000000000000e+00])

# Conversion: torch.quantize_per_tensor
# PyTorch supports explicit scale/zero_point and specific dtypes like quint2x4.
# TensorFlow's standard quantization (tf.quantization.quantize) infers scale/zp from min/max
# and typically uses quint8/qint8.
# To preserve the explicit scale/zero_point logic, we perform the calculation manually
# and cast to tf.quint8 (as quint2x4 is not available).
# Formula: q = round(x / scale) + zero_point
tensor_q1 = tf.cast(tf.round(data_fp / scale) + zero_point, tf.quint8)

input = [[tensor_q1, tensor_q1], {}, [], {}]

# Conversion: torch.equal
# torch.equal returns a boolean scalar.
# tf.equal returns a boolean tensor, so we use tf.reduce_all.
# input[0] unpacks to two tensors, input[1] is empty kwargs.
tf.reduce_all(tf.equal(*input[0]))
```