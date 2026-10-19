```python
import tensorflow as tf
import time

# Conversion: torch.manual_seed(0) -> tf.random.set_seed(0)
tf.random.set_seed(0)
shapes = [
     ((1, 12, 10, 64), (1, 12, 64, 10)),
     ((1, 12, 10, 10), (1, 12, 10, 64)),
]


# Conversion: PyTorch uses string "cpu"/"cuda", TensorFlow uses "/cpu:0"/"/gpu:0"
def benchmark_matmul(a_shape, b_shape, dtype=tf.float16, device="/cpu:0", repeat=500):
    # Conversion: torch.empty(...).uniform_(...) -> tf.random.uniform(...)
    # Note: TF uses functional API and device context managers instead of in-place ops
    with tf.device(device):
        A = tf.random.uniform(a_shape, minval=0.0, maxval=1.0, dtype=dtype) * 2 - 1
        B = tf.random.uniform(b_shape, minval=0.0, maxval=1.0, dtype=dtype) * 2 - 1
    # warm up
    for _ in range(5000):
        # Conversion: torch.matmul -> tf.matmul
        _ = tf.matmul(A, B)
    # run
    times = []
    for i in range(repeat):
        start = time.time()
        _ = tf.matmul(A, B)
        end = time.time()
        if i > 100:
            times.append(round((end - start) * 1000 * 1000))
    times.sort()
    print(times)
    avg_time_ms = sum(times) / len(times)
    return avg_time_ms

if __name__ == "__main__":
    for a_shape, b_shape in shapes:
        # Conversion: Passing TF-compatible device string
        t = benchmark_matmul(a_shape, b_shape, device="/cpu:0")
        print(f"{a_shape} x {b_shape}  ->  {t:.3f} us")
```