```python
import tensorflow as tf

# TensorFlow does not support explicit custom memory pools (MemPool) 
# or context-based memory pool switching like PyTorch.
# pool1 = make_custom_pool(1)

# with torch.cuda.use_mem_pool(torch.cuda.MemPool(pool1)):
print("Pool 1 ctx start")

# Use tf.device to ensure allocation on GPU, equivalent to device="cuda"
with tf.device('/GPU:0'):
    # torch.randn(8, device="cuda") -> tf.random.normal((8,))
    x1 = tf.random.normal((8,))

print("Pool 1 ctx end")
del x1
```