```python
import tensorflow as tf

# PyTorch: device='mps' is specific to Apple Silicon. TF uses default device or '/GPU:0'.
a = tf.zeros([1])
# the following line triggers the incorrect behavior, when commented, the remainder of the script appears to work as expected
# PyTorch: a.clamp(min=0.0) -> TF: tf.maximum(a, 0.0)
a_clamped = tf.maximum(a, 0.0)

b = tf.zeros([1])
print(b)
# PyTorch: b.clamp(min=1e-7) -> TF: tf.maximum(b, 1e-7)
c = tf.maximum(b, 1e-7)
print(c)

b = tf.zeros([1])
print(b)
# PyTorch: b.clamp(min=1e-7, max=None) -> TF: tf.maximum(b, 1e-7)
c = tf.maximum(b, 1e-7)
print(c)

b = tf.zeros([1])
print(b)
# PyTorch: b.clamp(min=1e-7, max=torch.inf) -> TF: tf.clip_by_value(b, 1e-7, float('inf'))
c = tf.clip_by_value(b, 1e-7, float('inf'))
print(c)

b = tf.zeros([1])
print(b)
# PyTorch: b.clamp_min(1e-7) -> TF: tf.maximum(b, 1e-7)
c = tf.maximum(b, 1e-7)
print(c)
```