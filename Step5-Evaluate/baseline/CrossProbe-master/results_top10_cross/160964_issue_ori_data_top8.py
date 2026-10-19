```python
import tensorflow as tf

# PyTorch: buckets = torch.arange(10, device="cuda")
# TensorFlow uses tf.range and explicit device placement
with tf.device('/GPU:0'):
    buckets = tf.range(10)
    # PyTorch: x = torch.tensor([2.5, 4.5, 5.5], device = "cuda")
    x = tf.constant([2.5, 4.5, 5.5])

# PyTorch: def test_fn(x): return torch.bucketize(x, buckets[1:])
# Note: torch.bucketize defaults to right=False, which corresponds to tf.searchsorted with side='left'
def test_fn(x):
    return tf.searchsorted(buckets[1:], x, side='left', out_type=tf.int64)

# PyTorch: compiled_fn = torch.compile(test_fn)
# TensorFlow uses tf.function to compile/trace the function into a graph
compiled_fn = tf.function(test_fn)

out_not_compiled = test_fn(x)
out_compiled = compiled_fn(x)
print(out_not_compiled)
print(out_compiled)
```