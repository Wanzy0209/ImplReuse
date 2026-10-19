```python
import tensorflow as tf

# PyTorch: torch.set_default_device('cuda')
# TensorFlow: Use a device context manager to place operations on GPU
with tf.device('/GPU:0'):
    # PyTorch: torch.randn(8192)
    # TensorFlow: tf.random.normal([8192])
    inp = tf.random.normal([8192])

    # PyTorch: func = torch.exp
    # TensorFlow: func = tf.math.exp
    func = tf.math.exp

    out1 = func(inp)

    # PyTorch: torch.compile(func)(inp)
    # TensorFlow: tf.function(func)(inp) is the equivalent for graph compilation/optimization
    out2 = tf.function(func)(inp)

    # PyTorch: func(inp.to(torch.float64))
    # TensorFlow: func(tf.cast(inp, tf.float64))
    out3_high = func(tf.cast(inp, tf.float64))

    # PyTorch: (out3_high - out1).abs().max()
    # TensorFlow: tf.reduce_max(tf.abs(out3_high - out1))
    print(tf.reduce_max(tf.abs(out3_high - out1)))
    print(tf.reduce_max(tf.abs(out3_high - out2)))
```