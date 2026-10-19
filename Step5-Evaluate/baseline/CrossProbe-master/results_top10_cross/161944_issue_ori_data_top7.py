```python
import tensorflow as tf

# PyTorch: torch.set_default_device('cuda')
# TensorFlow: Device placement is typically handled via context managers or automatic placement.
# We use a context manager here to ensure execution on GPU, mimicking the source intent.
with tf.device('/GPU:0'):
    inp = tf.random.normal([8192])
    func = tf.math.exp
    out1 = func(inp)

    # PyTorch: torch.compile(func)
    # TensorFlow: tf.function(func) compiles the function into a graph for optimization.
    out2 = tf.function(func)(inp)

    # PyTorch: inp.to(torch.float64)
    # TensorFlow: tf.cast(inp, tf.float64)
    out3_high = func(tf.cast(inp, tf.float64))

    # PyTorch: (out3_high - out1).abs().max()
    # TensorFlow: tf.reduce_max(tf.abs(...))
    print(tf.reduce_max(tf.abs(out3_high - out1)))
    print(tf.reduce_max(tf.abs(out3_high - out2)))
```