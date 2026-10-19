```python
import tensorflow as tf

print(tf.__version__, flush=True)
value = -3.0397450923919677734375
shape = (2, 5, 1, 9)
scale = 0.10856233049200226
zero_point = 31

# torch.full -> tf.fill
data_fp = tf.fill(shape, value)
data_fp = tf.cast(data_fp, tf.float32)

# torch.quantize_per_tensor
# Conversion: PyTorch quint2x4 is not supported in TF. Using quint8.
# Conversion: PyTorch uses scale/zero_point. TF uses min/max ranges.
# We calculate min/max from the tensor to perform quantization.
min_val = tf.reduce_min(data_fp)
max_val = tf.reduce_max(data_fp)

# tensor_q = tensor_q.to('cuda')
# Conversion: Move tensor to GPU using device context
with tf.device('/GPU:0'):
    tensor_q = tf.quantize(data_fp, min_val, max_val, tf.quint8)

input = [[tensor_q], {}, [], {}]

# torch.Tensor.tolist(*input[0],**input[1])
# Conversion: PyTorch tolist on quantized tensor returns dequantized values.
# We dequantize then convert to list.
# Unpacking input[0] gives tensor_q.
# Unpacking input[1] gives {}.
# So we call tolist on tensor_q.
# In TF: dequantize -> numpy -> tolist
tensor_deq = tf.dequantize(input[0][0], min_val, max_val, dtype=tf.float32)
tensor_deq.numpy().tolist()
```