```python
import tensorflow as tf

# TensorFlow optimizers typically have a single learning rate, unlike PyTorch's param_groups.
# We retrieve the current value and wrap it in a list to maintain the expected list[float] structure.
self._last_lr: list[float] = [tf.keras.backend.get_value(self.optimizer.learning_rate)]
```