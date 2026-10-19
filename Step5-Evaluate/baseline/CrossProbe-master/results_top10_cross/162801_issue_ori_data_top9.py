```python
import tensorflow as tf

# Print version
print(tf.__version__, flush=True)

shape = (3,)
scale = 0.08547895223232808
zero_point = 70

# Create tensor from list
data_fp = tf.constant([-5.98352670669555664062500000000000e+00, -5.72708988189697265625000000000000e+00, -5.72708988189697265625000000000000e+00])

# Conversion: torch.quantize_per_tensor
# PyTorch uses quint2x4 (2-bit unsigned). TF standard quantization uses quint8.
# We map to quint8 and derive min/max from scale/zero_point.
# Formula: zero_point = round(min / -scale) => min = -zero_point * scale
# Formula: scale = (max - min) / (qmax - qmin) => max = scale * (qmax - qmin) + min
# For quint8, qmin=0, qmax=255.
qmin = -zero_point * scale
qmax = (255 - zero_point) * scale

# tf.quantization.quantize returns (output, output_min, output_max)
quantized_tuple = tf.quantization.quantize(data_fp, qmin, qmax, tf.quint8)
tensor_q1 = quantized_tuple[0]

# Input structure
input = [[tensor_q1, tensor_q1], {}, [], {}]

# torch.equal(*input[0], **input[1]) -> tf.equal(tensor_q1, tensor_q1)
# Note: tf.equal returns a boolean tensor, not a scalar bool like torch.equal.
# To get a scalar bool, one would typically use tf.reduce_all(tf.equal(...)).numpy()
# However, preserving the direct call structure:
tf.equal(*input[0], **input[1])
```