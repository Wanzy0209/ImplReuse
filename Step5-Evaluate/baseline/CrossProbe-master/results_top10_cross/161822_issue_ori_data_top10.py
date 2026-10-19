```python
import tensorflow as tf
import time

warmup = 128
iters = 16384

# Conversion: Use tf.device to ensure execution on GPU, mimicking device='cuda'
with tf.device('/GPU:0'):
    # Conversion: torch.zeros -> tf.zeros, torch.bfloat16 -> tf.bfloat16
    a = tf.zeros((512, 512), dtype=tf.bfloat16)

    for _ in range(warmup):
        # Conversion: torch.matmul -> tf.matmul
        tf.matmul(a, a)

    # Conversion: TensorFlow eager execution is synchronous by default.
    # Unlike PyTorch, Python waits for the operation to complete, so explicit
    # synchronization like torch.cuda.synchronize() is not required.
    t0 = time.perf_counter()
    for _ in range(iters):
        tf.matmul(a, a)
    t1 = time.perf_counter()

print(f"{1e6 * (t1 - t0)/iters}")
```