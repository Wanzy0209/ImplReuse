```python
import tensorflow as tf

# Use GPU context to match device="cuda"
with tf.device('/GPU:0'):
    # torch.arange(10) -> tf.range(10)
    buckets = tf.range(10)
    # torch.tensor(...) -> tf.constant(...)
    x = tf.constant([2.5, 4.5, 5.5])

def test_fn(x):
    # torch.bucketize(x, boundaries) is equivalent to tf.searchsorted(boundaries, x)
    # Note: Argument order is swapped in TF compared to PyTorch.
    # We specify out_type=tf.int64 to match PyTorch's default return type.
    return tf.searchsorted(buckets[1:], x, out_type=tf.int64)

# torch.compile -> tf.function
# tf.function converts a Python function into a TensorFlow graph for optimization
compiled_fn = tf.function(test_fn)

out_not_compiled = test_fn(x)
out_compiled = compiled_fn(x)
print(out_not_compiled)
print(out_compiled)
```