```python
import tensorflow as tf

print(tf.__version__, flush=True)

value = -3.0397450923919677734375
shape = (2, 5, 1, 9)
scale = 0.10856233049200226
zero_point = 31

# torch.full(shape, value, dtype=torch.float32, device='cpu')
# Create tensor on CPU explicitly
with tf.device('/CPU:0'):
    data_fp = tf.fill(shape, value)

# torch.quantize_per_tensor(data_fp, scale=scale, zero_point=zero_point, dtype=torch.quint2x4)
# tensor_q = tensor_q.to('cuda')
# TensorFlow does not natively support quint2x4 (2-bit quantization).
# We simulate the quantization logic: q = round(f / scale) + zero_point
# and clip to the 2-bit range [0, 3].
# We also place the computation on GPU to mimic .to('cuda').
with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
    q_values = tf.round(data_fp / scale) + zero_point
    q_values = tf.cast(q_values, tf.int32)
    tensor_q = tf.clip_by_value(q_values, 0, 3)

input = [[tensor_q], {}, [], {}]

# torch.Tensor.tolist(*input[0], **input[1])
# Equivalent to tensor_q.tolist()
# In TensorFlow, we convert to numpy array then to list
input[0][0].numpy().tolist()
```