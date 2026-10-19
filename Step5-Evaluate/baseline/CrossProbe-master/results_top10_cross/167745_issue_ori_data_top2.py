```python
import tensorflow as tf

# make_custom_pool is assumed to be a user-defined function
pool1 = make_custom_pool(1)

# Conversion: TensorFlow does not support explicit custom memory pool switching via a context manager.
# We use tf.device to ensure GPU placement, which is the functional equivalent for device allocation.
with tf.device('/GPU:0'):
    print("Pool 1 ctx start")
    # Conversion: torch.randn generates random numbers from a normal distribution.
    # The TensorFlow equivalent is tf.random.normal.
    x1 = tf.random.normal([8])
    print("Pool 1 ctx end")
del x1
```