```python
import tensorflow as tf
import time

# Conversion: torch.manual_seed(0) -> tf.random.Generator.from_seed(0)
# Using the Generator API as suggested by the context 'stateful_random_ops.from_seed'
g = tf.random.Generator.from_seed(0)

shapes = [
     ((1, 12, 10, 64), (1, 12, 64, 10)),
     ((1, 12, 10, 10), (1, 12, 10, 64)),
]


def benchmark_matmul(a_shape, b_shape, dtype=tf.float16, device="cpu", repeat=500):
    # Conversion: torch.empty(...).uniform_(0,1) * 2 - 1 -> tf.random.uniform(..., minval=-1.0, maxval=1.0)
    # Note: torch.empty allocates uninitialized memory, but here it is immediately filled.
    # tf.random.uniform creates the tensor directly with the specified distribution.
    with tf.device(device):
        A = g.uniform(shape=a_shape, dtype=dtype, minval=-1.0, maxval=1.0)
        B = g.uniform(shape=b_shape, dtype=dtype, minval=-1.0, maxval=1.0)
        
        # warm up
        for _ in range(5000):
            # Conversion: torch.matmul -> tf.matmul
            # Note: The provided context suggested 'transpose_v2', but that is incorrect for matrix multiplication.
            # We use tf.matmul to preserve the original logic.
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
        t = benchmark_matmul(a_shape, b_shape)
        print(f"{a_shape} x {b_shape}  ->  {t:.3f} us")
```