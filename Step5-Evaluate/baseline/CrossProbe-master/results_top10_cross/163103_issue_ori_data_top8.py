```python
import tensorflow as tf

# TensorFlow optimizers typically have a single learning rate rather than param_groups.
# We extract the current learning rate value and wrap it in a list to match the expected structure.
self._last_lr: list[float] = [tf.keras.backend.get_value(self.optimizer.learning_rate)]
```