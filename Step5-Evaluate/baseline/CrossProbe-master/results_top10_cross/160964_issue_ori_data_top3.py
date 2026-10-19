```python
import tensorflow as tf

# Use GPU context to match device="cuda"
with tf.device('/GPU:0'):
    # torch.arange(10, device="cuda") -> tf.range(10)
    buckets = tf.range(10)
    # torch.tensor([2.5, 4.5, 5.5], device = "cuda") -> tf.constant([2.5, 4.5, 5.5])
    x = tf.constant([2.5, 4.5, 5.5])

def test_fn(x):
    # torch.bucketize(x, buckets[1:]) -> tf.searchsorted(boundaries, x, side='right')
    # PyTorch bucketize with default right=True corresponds to side='right' in TensorFlow
    return tf.searchsorted(buckets[1:], x, side='right')

# torch.compile(test_fn) -> tf.function(test_fn)
compiled_fn = tf.function(test_fn)

out_not_compiled = test_fn(x)
out_compiled = compiled_fn(x)
print(out_not_compiled)
print(out_compiled)
```