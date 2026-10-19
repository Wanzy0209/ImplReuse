```python
import tensorflow as tf

# Print version
print(tf.__version__, flush=True)

shape = (3,)
scale = 0.08547895223232808
zero_point = 70

# Create float tensor
data_fp = tf.constant([-5.98352670669555664062500000000000e+00, -5.72708988189697265625000000000000e+00, -5.72708988189697265625000000000000e+00])

# Conversion: PyTorch quint2x4 -> TensorFlow quint8 (standard quantized type)
# Calculate min/max for TF quantization from scale/zero_point
# Formula: real_value = scale * (quantized_value - zero_point)
# For quint8, range is [0, 255]
qmin = 0
qmax = 255
min_val = scale * (qmin - zero_point)
max_val = scale * (qmax - zero_point)

# Quantize the tensor
# tf.quantization.quantize returns (output, output_min, output_max)
tensor_q1, _, _ = tf.quantization.quantize(data_fp, min_val, max_val, tf.quint8)

input = [[tensor_q1, tensor_q1], {}, [], {}]

# PyTorch torch.equal returns a single boolean scalar.
# TensorFlow tf.equal returns a boolean tensor. We use reduce_all to match behavior.
# Note: input[1] is an empty dict, so **input[1] expands to nothing.
tf.reduce_all(tf.equal(*input[0], **input[1]))
```