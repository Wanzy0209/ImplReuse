```python
import tensorflow as tf

# TensorFlow optimizers typically have a single learning rate rather than parameter groups.
# We use get_value to retrieve the scalar value from the tensor/variable.
self._last_lr: list[float] = [tf.keras.backend.get_value(self.optimizer.learning_rate)]
```