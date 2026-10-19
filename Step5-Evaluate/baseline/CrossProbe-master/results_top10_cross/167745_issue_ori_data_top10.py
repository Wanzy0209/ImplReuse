```python
import tensorflow as tf

# Conversion: TensorFlow does not support custom CUDA memory pools via public API.
# The pool creation is commented out as it has no direct equivalent.
# pool1 = make_custom_pool(1)

# Conversion: Replaced torch.cuda.use_mem_pool with tf.device to manage allocation context.
# with torch.cuda.use_mem_pool(torch.cuda.MemPool(pool1)):
with tf.device('/GPU:0'):
    print("Pool 1 ctx start")
    # Conversion: torch.randn(8, device="cuda") -> tf.random.normal((8,))
    x1 = tf.random.normal((8,))
    print("Pool 1 ctx end")
del x1
```