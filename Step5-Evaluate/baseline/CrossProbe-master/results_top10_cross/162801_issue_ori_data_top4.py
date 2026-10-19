```python
import tensorflow as tf
print(tf.__version__, flush=True)

shape = (3,)
scale = 0.08547895223232808
zero_point = 70
# Conversion: torch.tensor -> tf.constant
data_fp = tf.constant([-5.98352670669555664062500000000000e+00, -5.72708988189697265625000000000000e+00, -5.72708988189697265625000000000000e+00])

# Conversion: torch.quantize_per_tensor with explicit scale/zero_point and quint2x4 (4-bit)
# TensorFlow does not have a native 4-bit eager tensor type. We simulate the quantization logic manually.
# Formula: q = round(x / scale) + zero_point
# quint2x4 range is [0, 15]
tensor_q1 = tf.cast(
    tf.clip_by_value(
        tf.round(data_fp / scale) + zero_point, 
        0.0, 
        15.0
    ), 
    tf.uint8
)

input = [[tensor_q1, tensor_q1], {}, [], {}]

# Conversion: torch.equal returns a single boolean, tf.equal returns a tensor of booleans.
# We use reduce_all to get a single scalar boolean tensor.
tf.reduce_all(tf.equal(*input[0], **input[1]))
```