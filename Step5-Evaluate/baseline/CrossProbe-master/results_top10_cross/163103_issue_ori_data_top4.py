```python
import tensorflow as tf

# Conversion Note: PyTorch optimizers use 'param_groups' to support multiple learning rates.
# TensorFlow optimizers typically have a single 'learning_rate' attribute (which can be a float, variable, or schedule).
# We wrap the single learning rate in a list to preserve the list structure expected by the rest of the logic.
self._last_lr: list[float] = [self.optimizer.learning_rate]
```