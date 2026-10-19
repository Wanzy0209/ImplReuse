```python
import tensorflow as tf

# Conversion Note: TensorFlow manages memory automatically and does not support 
# custom memory pools via context managers like PyTorch's use_mem_pool.
# Dummy implementations are provided below to preserve code structure and validity.
class MemPool:
    def __init__(self, pool):
        pass

def use_mem_pool(pool):
    class _MemPoolContext:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    return _MemPoolContext()

def make_custom_pool(val):
    return val

pool1 = make_custom_pool(1)

with use_mem_pool(MemPool(pool1)):
    print("Pool 1 ctx start")
    # Conversion Note: torch.randn(8, device="cuda") translates to tf.random.normal((8,))
    # TensorFlow automatically uses the GPU if available.
    x1 = tf.random.normal((8,))
    print("Pool 1 ctx end")
del x1
```