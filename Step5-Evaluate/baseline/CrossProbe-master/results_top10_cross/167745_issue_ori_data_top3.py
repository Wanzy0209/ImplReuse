```python
import tensorflow as tf

# Conversion: PyTorch's custom memory pools (torch.cuda.use_mem_pool) are not supported in TensorFlow.
# TensorFlow manages memory automatically via the BFC allocator. We omit the pool creation and context manager.
# pool1 = make_custom_pool(1)

# with torch.cuda.use_mem_pool(torch.cuda.MemPool(pool1)):
print("Pool 1 ctx start")
# Conversion: torch.randn(8, device="cuda") creates a random tensor on the GPU.
# In TensorFlow, tf.random.normal creates a tensor with values from a normal distribution.
# Device placement is handled automatically (GPU if available).
x1 = tf.random.normal((8,))
print("Pool 1 ctx end")

del x1
```