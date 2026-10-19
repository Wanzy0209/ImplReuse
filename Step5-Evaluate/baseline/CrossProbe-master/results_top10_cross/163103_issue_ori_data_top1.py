```python
import tensorflow as tf

# Conversion Note: PyTorch optimizers use param_groups to manage multiple learning rates.
# TensorFlow optimizers typically have a single learning_rate attribute.
# We wrap it in a list to preserve the list structure expected by the original logic.
self._last_lr: list[float] = [self.optimizer.learning_rate]
```