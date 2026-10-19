```python
import tensorflow as tf

# Conversion: TensorFlow manages memory automatically; custom memory pools are not exposed in the public API.
# pool1 = make_custom_pool(1)

# Conversion: Replaced memory pool context with GPU device context
# with torch.cuda.use_mem_pool(torch.cuda.MemPool(pool1)):
with tf.device('/GPU:0'):
    print("Pool 1 ctx start")
    # Conversion: torch.randn(8, device="cuda") -> tf.random.normal([8])
    x1 = tf.random.normal([8])
    print("Pool 1 ctx end")
del x1
```