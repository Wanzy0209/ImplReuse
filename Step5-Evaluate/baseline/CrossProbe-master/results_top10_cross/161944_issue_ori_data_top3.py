```python
import tensorflow as tf

# torch.set_default_device('cuda')
# In TensorFlow, device placement is typically handled via context managers
with tf.device('/GPU:0'):
    # inp = torch.randn(8192)
    inp = tf.random.normal((8192,))

    # func = torch.exp
    func = tf.exp

    # out1 = func(inp)
    out1 = func(inp)

    # out2 = torch.compile(func)(inp)
    # torch.compile optimizes the function; tf.function is the TensorFlow equivalent
    out2 = tf.function(func)(inp)

    # out3_high = func(inp.to(torch.float64))
    out3_high = func(tf.cast(inp, tf.float64))

    # print((out3_high - out1).abs().max())
    print(tf.reduce_max(tf.abs(out3_high - out1)))

    # print((out3_high - out2).abs().max())
    print(tf.reduce_max(tf.abs(out3_high - out2)))
```