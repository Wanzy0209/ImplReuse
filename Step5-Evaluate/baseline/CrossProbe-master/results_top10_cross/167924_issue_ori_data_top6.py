```python
import tensorflow as tf

# Note: PyTorch's 'mps' device is specific to Apple Silicon.
# In TensorFlow, device placement is typically automatic or managed via context managers (e.g., tf.device('/GPU:0')).
# We create tensors here without the explicit 'mps' constraint.
counts = tf.constant([0, 1, 0])
data = tf.range(2)

# PyTorch's repeat_interleave maps to tf.repeat in TensorFlow.
# counts[1:3] selects the slice [1, 0].
# data [0, 1] repeated [1, 0] times results in [0].
tf.repeat(data, counts[1:3], axis=0)
```