```python
import tensorflow as tf

# Use GPU device to match "cuda" requirement
with tf.device('/GPU:0'):
    # torch.arange(10, device="cuda")
    buckets = tf.range(10)
    # torch.tensor([2.5, 4.5, 5.5], device = "cuda")
    x = tf.constant([2.5, 4.5, 5.5])

def test_fn(x):
    # torch.bucketize(x, buckets[1:]) translates to tf.searchsorted
    # Note: PyTorch bucketize returns int64 by default, so we specify out_type
    return tf.searchsorted(buckets[1:], x, out_type=tf.int64)

# torch.compile translates to tf.function
compiled_fn = tf.function(test_fn)

out_not_compiled = test_fn(x)
out_compiled = compiled_fn(x)
print(out_not_compiled)
print(out_compiled)
```