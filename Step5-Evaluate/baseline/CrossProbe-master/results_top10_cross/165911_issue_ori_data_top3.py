```python
import tensorflow as tf

# Conversion: torch.nn.functional.linear(x, w) -> tf.linalg.matmul(x, w, transpose_b=True)
def compute(x, w):
    return tf.linalg.matmul(x, w, transpose_b=True)

# Conversion: torch._check -> tf.debugging.assert_equal
# Conversion: torch.empty_like -> tf.empty_like
def nop(x, w):
    tf.debugging.assert_equal(tf.shape(x)[0], 0)
    return tf.empty_like(x)  # whatever the output size should be

# Conversion: torch.cond -> tf.cond
def chunked_compute(x, w):
    sz = tf.shape(x)[0]
    tf.debugging.assert_less_equal(sz, 8)
    # Note: tf.cond functions take no arguments, so we use lambdas to pass arguments
    out0 = tf.cond(sz > 0, lambda: compute(x[0:2], w), lambda: nop(x[0:2], w))
    out1 = tf.cond(sz > 2, lambda: compute(x[2:4], w), lambda: nop(x[2:4], w))
    out2 = tf.cond(sz > 4, lambda: compute(x[4:6], w), lambda: nop