```python
import tensorflow as tf

print(tf.__version__, flush=True)

value = -3.0397450923919677734375
shape = (2, 5, 1, 9)
scale = 0.10856233049200226
zero_point = 31

# torch.full(shape, value, dtype=torch.float32, device='cpu')
with tf.device('/CPU:0'):
    data_fp = tf.fill(shape, tf.cast(value, tf.float32))

# torch.quantize_per_tensor(data_fp, scale=scale, zero_point=zero_point, dtype=torch.quint2x4)
# Conversion: torch.quint2x4 is not a standard public dtype. Assuming qint8 due to negative value.
# TF quantization requires min/max. Deriving from scale/zero_point for qint8 range [-128, 127].
qmin = -128
qmax = 127
input_min = (qmin - zero_point) * scale
input_max = (qmax - zero_point) * scale

tensor_q, _, _ = tf.quantization.quantize(
    data_fp,
    input_min,
    input_max,
    T=tf.qint8,
    mode='MIN_COMBINED'
)

# tensor_q = tensor_q.to('cuda')
with tf.device('/GPU:0'):
    tensor_q = tf.identity(tensor_q)

input = [[tensor_q], {}, [], {}]

# torch.Tensor.tolist(*input[0], **input[1])
# PyTorch QTensor.tolist() returns dequantized values.
# We dequantize then convert to list.
tensor_deq, _, _ = tf.quantization.dequantize(
    tensor_q,
    input_min,
    input_max,
    mode='MIN_COMBINED'
)

# Replicate the call structure
input[0][0].numpy().tolist()
```