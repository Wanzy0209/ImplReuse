```python
import tensorflow as tf

# Conversion Note: PyTorch optimizers support parameter groups with distinct learning rates.
# TensorFlow Keras optimizers typically have a single learning_rate attribute.
# We retrieve the current scalar value of the learning rate and wrap it in a list
# to maintain compatibility with the expected list[float] structure.
self._last_lr: list[float] = [tf.keras.backend.get_value(self.optimizer.learning_rate)]
```