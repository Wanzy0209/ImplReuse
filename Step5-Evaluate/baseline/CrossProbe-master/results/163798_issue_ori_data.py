```python
import tensorflow as tf

# torch.compile with backend="eager" runs in eager mode.
# TensorFlow executes eagerly by default, so no decorator is needed.
def func(a):
    # Convert tensor to a Python list and unpack values
    u0, u1 = a.numpy().tolist()
    return a * u0 * u1

func(tf.constant([1, 2]))
```