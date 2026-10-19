```python
import tensorflow as tf
import time

warmup = 128
iters = 16384

# Conversion: torch.zeros with device='cuda' -> tf.zeros inside tf.device context
# Conversion: dtype=torch.bfloat16 -> dtype=tf.bfloat16
with tf.device('/GPU:0'):
    a = tf.zeros((512, 512), dtype=tf.bfloat16)

    for _ in range(warmup):
        # Conversion: torch.matmul -> tf.matmul
        # Assigning to _ to ensure the operation is executed and not optimized out
        _ = tf.matmul(a, a)

    # Conversion: torch.cuda.synchronize()
    # TensorFlow eager execution is asynchronous. We force synchronization
    # by querying device memory info, which blocks until the GPU is idle.
    tf.config.experimental.get_memory_info('GPU:0')

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = tf.matmul(a, a)
    tf.config.experimental.get_memory_info('GPU:0')
    t1 = time.perf_counter()

print(f"{1e6 * (t1 - t0)/iters}")
```