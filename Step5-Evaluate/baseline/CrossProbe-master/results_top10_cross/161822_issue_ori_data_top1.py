```python
import tensorflow as tf
import time

warmup = 128
iters = 16384

# Conversion: torch.zeros with device='cuda' and dtype=torch.bfloat16
# Using tf.device context to ensure GPU placement
with tf.device('/GPU:0'):
    a = tf.zeros((512, 512), dtype=tf.bfloat16)

    for _ in range(warmup):
        # Conversion: torch.matmul -> tf.matmul
        tf.matmul(a, a)

    # Conversion: torch.cuda.synchronize()
    # Note: TensorFlow eager execution is asynchronous. There is no direct public 
    # equivalent to torch.cuda.synchronize() to force GPU completion in eager mode.
    # We use a pass statement to preserve structure, but timing will include dispatch overhead.
    pass

    t0 = time.perf_counter()
    for _ in range(iters):
        tf.matmul(a, a)
    
    # Conversion: torch.cuda.synchronize()
    pass

    t1 = time.perf_counter()
    print(f"{1e6 * (t1 - t0)/iters}")
```