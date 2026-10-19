```python
import tensorflow as tf
print(tf.__version__, flush=True)

shape = (3,)
scale = 0.08547895223232808
zero_point = 70
data_fp = tf.constant([-5.98352670669555664062500000000000e+00, -5.72708988189697265625000000000000e+00, -5.72708988189697265625000000000000e+00])

# Conversion: PyTorch quint2x4 is a 2-bit unsigned integer type (range 0-3).
# TensorFlow quantization typically uses uint8. We calculate the min/max float values
# corresponding to the PyTorch quantization range to preserve the logic.
q_min = 0
q_max = 3
min_val = (q_min - zero_point) * scale
max_val = (q_max - zero_point) * scale

# tf.quantization.quantize returns (output, min_out, max_out)
tensor_q1 = tf.quantization.quantize(
    data_fp,
    min_val=min_val,
    max_val=max_val,
    T=tf.uint8,
    mode='MIN_COMBINED',
    round_mode='HALF_AWAY_FROM_ZERO'
)[0]

input = [[tensor_q1, tensor_q1], {}, [], {}]

# Conversion: torch.equal returns a single boolean scalar.
# tf.equal returns a boolean tensor, so we use reduce_all to match the behavior.
tf.reduce_all(tf.equal(*input[0], **input[1]))
```