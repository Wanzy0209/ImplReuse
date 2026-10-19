```python
import tensorflow as tf

# TensorFlow does not have a direct equivalent for custom memory pools (torch.cuda.MemPool).
# We assume make_custom_pool is a user-defined function and omit the pool logic.
# pool1 = make_custom_pool(1)

# Replacing torch.cuda.use_mem_pool with tf.device to ensure GPU execution
with tf.device('/GPU:0'):
    print("Pool 1 ctx start")
    # Conversion: torch.randn(8, device="cuda") -> tf.random.normal(shape=[8])
    x1 = tf.random.normal(shape=[8])
    print("Pool 1 ctx end")
del x1
```