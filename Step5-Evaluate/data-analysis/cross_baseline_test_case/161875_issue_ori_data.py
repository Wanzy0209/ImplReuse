```python
import tensorflow as tf

# Conversion: PyTorch input shape (Batch, Channels, Length) -> TF (Batch, Length, Channels)
input_data = tf.random.normal(shape=(1, 32, 3))

# Conversion: LazyConv1d infers input channels. TF Conv1D infers input shape on first call.
lazy_conv1d = tf.keras.layers.Conv1D(filters=16, kernel_size=3, strides=1,
    padding=9223372036854775803, use_bias=True)

# Conversion: PyTorch .to(device) moves module. TF uses device context for execution.
with tf.device('/CPU:0'):
    output = lazy_conv1d(input_data)
```