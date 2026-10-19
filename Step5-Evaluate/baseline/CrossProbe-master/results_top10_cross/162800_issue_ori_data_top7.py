```python
import tensorflow as tf

print(tf.__version__, flush=True)

value = -3.0397450923919677734375
shape = (2, 5, 1, 9)
scale = 0.10856233049200226
zero_point = 31

# torch.full creates a tensor filled with the specified value
data_fp = tf.fill(shape, value)
data_fp = tf.cast(data_fp, tf.float32) # Ensure dtype matches torch.float32

# torch.quantize_per_tensor
# TensorFlow does not natively support quint2x4 (2-bit quantization).
# We map this to quint8 (8-bit quantization) as the closest standard equivalent.
# To preserve the scale and zero_point, we calculate the corresponding min/max range
# for the quint8 data type (range 0-255).
qmin = 0
qmax = 255
min_range = scale * (qmin - zero_point)
max_range = scale * (qmax - zero_point)

# Perform quantization
# tf.quantization.quantize returns (quantized_tensor, min_tensor, max_tensor)
tensor_q, min_t, max_t = tf.quantization.quantize(data_fp, min_range, max_range, tf.quint8)

# tensor_q.to('cuda')
# In TensorFlow, device placement is handled via context managers or explicit placement.
# We use tf.identity within a device block to move the tensor to GPU.
with tf.device('/GPU:0'):
    tensor_q = tf.identity(tensor_q)

input_list = [[tensor_q], {}, [], {}]

# torch.Tensor.tolist
# PyTorch's tolist() on a quantized tensor returns the dequantized float values.
# We must dequantize the TensorFlow tensor before converting to a list.
# Unpacking input: *input_list[0] passes tensor_q, **input_list[1] passes {}.
dequantized_tensor = tf.dequantize(tensor_q, min_t, max_t)
dequantized_tensor.numpy().tolist()
```