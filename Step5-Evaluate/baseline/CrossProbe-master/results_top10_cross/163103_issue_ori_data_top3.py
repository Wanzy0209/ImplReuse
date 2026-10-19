```python
import tensorflow as tf

# Conversion Note: PyTorch optimizers manage parameter groups (list of dicts), each with a specific 'lr'.
# Keras optimizers typically have a single 'learning_rate' property. We extract the current value
# and wrap it in a list to maintain the expected list[float] structure.
self._last_lr: list[float] = [tf.keras.backend.get_value(self.optimizer.learning_rate)]
```