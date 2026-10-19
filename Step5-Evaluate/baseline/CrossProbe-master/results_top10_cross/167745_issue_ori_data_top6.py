```python
import tensorflow as tf

# TensorFlow does not support custom memory pools (MemPool) like PyTorch.
# The following line is commented out as the functionality is not available.
# pool1 = make_custom_pool(1)

# Replaced torch.cuda.use_mem_pool with tf.device to ensure GPU execution.
# Custom memory pool management is handled internally by TensorFlow.
with tf.device('/GPU:0'):
    print("Pool 1 ctx start")
    # torch.randn(8, device="cuda") -> tf.random.normal((8,))
    x1 = tf.random.normal((8,))
    print("Pool 1 ctx end")
del x1
```