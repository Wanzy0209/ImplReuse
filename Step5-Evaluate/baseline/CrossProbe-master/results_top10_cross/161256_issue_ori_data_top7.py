```python
import tensorflow as tf

# torch.set_default_dtype(torch.float64)
# Note: TensorFlow does not have a global default dtype setting. 
# We explicitly specify dtype=tf.float64 in the operations below to match the behavior.
# We use a device context to place tensors on the GPU, equivalent to device="cuda".

with tf.device('/GPU:0'):
    # torch.randn(1000, 1000, device="cuda")
    # Conversion: tf.random.normal generates values from a standard normal distribution.
    x = tf.random.normal((1000, 1000), dtype=tf.float64)
    y = tf.random.normal((1000, 1000), dtype=tf.float64)

    # z = x @ y
    # Conversion: Matrix multiplication is supported via the @ operator in TensorFlow 2.x.
    z = x @ y

    # print(z[0, 0])
    print(z[0, 0])
```