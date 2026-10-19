```python
import tensorflow as tf

# Use GPU context to match device="cuda"
with tf.device('/GPU:0'):
    # torch.arange(10, device="cuda") -> tf.range(10)
    buckets = tf.range(10)
    # torch.tensor([2.5, 4.5, 5.5], device = "cuda") -> tf.constant([2.5, 4.5, 5.5])
    x = tf.constant([2.5, 4.5, 5.5])

def test_fn(x):
    # torch.bucketize(x, boundaries) -> tf.searchsorted(sorted_sequence, values)
    # Note: PyTorch bucketize defaults to right=False, which corresponds to side='left' in TF.
    # The argument order is also swapped in TensorFlow.
    return tf.searchsorted(buckets[1:], x)

# torch.compile(test_fn) -> tf.function(test_fn)
# tf.function creates a callable that compiles the function into a TensorFlow graph.
compiled_fn = tf.function(test_fn)

out_not_compiled = test_fn(x)
out_compiled = compiled_fn(x)
print(out_not_compiled)
print(out_compiled)
```