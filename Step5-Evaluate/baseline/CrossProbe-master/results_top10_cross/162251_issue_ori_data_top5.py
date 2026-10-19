```python
import tensorflow as tf

print(tf.__version__)
# TensorFlow does not support complex32 (float16 real/imag), mapping to complex64
t = tf.zeros((5, 5, 9, 3), dtype=tf.complex64)
input = [
    [545460846592],
    {},
    [t],
    {}
]
# Note: tf.keras.layers.PixelShuffle defaults to channels_last (NHWC), unlike PyTorch (NCHW)
r1 = tf.keras.layers.PixelShuffle(*input[0], **input[1])
r2 = r1(*input[2], **input[3])
```