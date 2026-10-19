```python
import tensorflow as tf

# Note: TensorFlow uses GPU by default if available. Explicit device placement requires a context manager (e.g., with tf.device('/GPU:0')).
# torch.ones creates a tensor of ones, equivalent to tf.ones.
# .item() extracts a standard Python scalar, equivalent to .numpy().item() in TensorFlow.
print(tf.ones(1).numpy().item())
```