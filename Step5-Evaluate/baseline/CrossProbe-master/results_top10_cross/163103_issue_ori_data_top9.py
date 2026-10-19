```python
# TensorFlow optimizers typically use a single learning rate instead of parameter groups.
# We wrap the learning rate in a list to preserve the list[float] type hint.
self._last_lr: list[float] = [self.optimizer.learning_rate]
```