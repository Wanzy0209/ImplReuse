```python
import tensorflow as tf

# Conversion Note: TensorFlow manages memory automatically via the BFC allocator.
# There is no direct equivalent to torch.cuda.MemPool or custom memory pools in the Python API.
# pool1 = make_custom_pool(1)

# Conversion Note: Using tf.device to preserve the context manager structure.
# In TensorFlow, specific memory pools cannot be selected manually in this manner.
with tf.device('/GPU:0'):
    print("Pool 1 ctx start")
    # torch.randn(8, device="cuda") -> tf.random.normal((8,))
    x1 = tf.random.normal((8,))
    print("Pool 1 ctx end")
del x1
```