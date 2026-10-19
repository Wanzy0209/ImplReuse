```python
import tensorflow as tf
import time

warmup = 128
iters = 16384

# Conversion: Explicitly place operations on GPU to match 'device="cuda"'
with tf.device('/GPU:0'):
    # Conversion: torch.zeros -> tf.zeros
    # Conversion: dtype=torch.bfloat16 -> dtype=tf.bfloat16
    a = tf.zeros((512, 512), dtype=tf.bfloat16)

    for _ in range(warmup):
        # Conversion: torch.matmul -> tf.matmul
        tf.matmul(a, a)

    # Conversion: torch.cuda.synchronize()
    # TensorFlow eager execution is asynchronous. We force synchronization
    # by querying device memory info, which acts as a barrier.
    tf.config.experimental.get_memory_info('GPU:0')

    t0 = time.perf_counter()
    for _ in range(iters):
        tf.matmul(a, a)
    
    # Conversion: torch.cuda.synchronize()
    tf.config.experimental.get_memory_info('GPU:0')
    
    t1 = time.perf_counter()
    print(f"{1e6 * (t1 - t0)/iters}")
```