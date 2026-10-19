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
    tf.matmul(a, a)

# Conversion: torch.cuda.synchronize() is not required in TensorFlow eager mode
# as operations execute synchronously by default.
t0 = time.perf_counter()
for _ in range(iters):
    # Conversion: torch.matmul -> tf.matmul
    tf.matmul(a, a)
# Conversion: torch.cuda.synchronize() is not required in TensorFlow eager mode
t1 = time.perf_counter()
print(f"{1e6 * (t1 - t0)/iters}")
```