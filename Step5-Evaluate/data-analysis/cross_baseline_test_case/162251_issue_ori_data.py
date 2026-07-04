```python
import tensorflow as tf
print(tf.__version__)
# Conversion: torch.complex32 is not available in TF, using tf.complex64
t = tf.zeros((5, 5, 9, 3), dtype=tf.complex64)
input = [
    [545460846592],
    {},
    [t],
    {}
]
# Conversion: torch.nn.PixelShuffle -> tf.keras.layers.PixelShuffle
r1 = tf.keras.layers.PixelShuffle(*input[0],**input[1])
r2 = r1(*input[2],**input[3])
```