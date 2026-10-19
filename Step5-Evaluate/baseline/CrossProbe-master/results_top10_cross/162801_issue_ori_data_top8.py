```python
import tensorflow as tf
print(tf.__version__, flush=True)

shape = (3,)
scale = 0.08547895223232808
zero_point = 70
data_fp = tf.constant([-5.98352670669555664062500000000000e+00, -5.72708988189697265625000000000000e+00, -5.72708988189697265625000000000000e+00])

# Conversion: PyTorch's quint2x4 is a packed 2-bit format not directly available in TensorFlow.
# We simulate the quantization logic manually using the provided scale and zero_point,
# then cast to quint8 as the closest standard quantized type.
tensor_q1 = tf.cast(tf.round(data_fp / scale + zero_point), tf.quint8)

input = [[tensor_q1, tensor_q1], {}, [], {}]
# Conversion: torch.equal returns a single boolean, while tf.equal returns a boolean tensor.
# We use tf.reduce_all to mimic the scalar boolean result of torch.equal.
tf.reduce_all(tf.equal(*input[0]))
```