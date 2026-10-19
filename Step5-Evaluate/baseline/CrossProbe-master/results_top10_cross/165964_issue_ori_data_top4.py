```python
import tensorflow as tf

# Use device context to place tensor on GPU, equivalent to device="cuda"
with tf.device('/GPU:0'):
    # tf.ones creates the tensor, .numpy() converts to numpy array, .item() extracts the scalar value
    print(tf.ones(1).numpy().item())
```