```python
import tensorflow as tf

# Conversion: PyTorch optimizers use param_groups to store learning rates for different parameter sets.
# TensorFlow optimizers typically have a single learning_rate attribute.
# We wrap the single learning rate in a list to maintain the list[float] type structure.
self._last_lr: list[float] = [self.optimizer.learning_rate]
```