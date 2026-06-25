```python
import tensorflow as tf

# Print TensorFlow version
print(tf.__version__, flush=True)

shape = (3,)
scale = 0.08547895223232808
zero_point = 70

# Conversion: torch.tensor -> tf.constant
data_fp = tf.constant([-5.98352670669555664062500000000000e+00, -5.72708988189697265625000000000000e+00, -5.72708988189697265625000000000000e+00])

# Conversion: torch.quantize_per_tensor
# PyTorch uses quint2x4 (2-bit). TF standard quantization uses quint8 (8-bit).
# We map to quint8 and calculate min/max from scale/zero_point.
# min = scale * (0 - zero_point), max = scale * (255 - zero_point)
min_val = scale * (0 - zero_point)
max_val = scale * (255 - zero_point)
# tf.quantization.quantize returns (output, output_min, output_max)
tensor_q1, _, _ = tf.quantization.quantize(data_fp, min_val, max_val, tf.quint8)

input = [[tensor_q1, tensor_q1], {}, [], {}]

# Conversion: torch.equal -> tf.equal
# Note: torch.equal returns a single bool, tf.equal returns a boolean tensor.
# The input structure unpacks to tf.equal(tensor_q1, tensor_q1).
tf.equal(input[0][0], input[0][1])
```