```python
import tensorflow as tf
import contextlib

# NOTE: PyTorch's custom memory pool management (torch.cuda.MemPool) is not
# directly supported in TensorFlow's public API. TensorFlow handles memory
# allocation automatically via the BFC allocator. We use a null context
# to preserve the code structure.

# pool1 = make_custom_pool(1) # No direct equivalent in TF
pool1 = None

# with torch.cuda.use_mem_pool(torch.cuda.MemPool(pool1)):
with contextlib.nullcontext(): # Conversion: Memory pool context not available in TF
    print("Pool 1 ctx start")
    # Conversion: torch.randn(8, device="cuda") -> tf.random.normal
    # TF uses GPU by default if available. Explicit device placement can be done with tf.device("/GPU:0").
    x1 = tf.random.normal((8,), dtype=tf.float32)
    print("Pool 1 ctx end")
del x1
```