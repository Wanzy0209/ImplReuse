```python
import tensorflow as tf
import time

# torch.manual_seed(0) -> tf.random.set_seed(0)
tf.random.set_seed(0)

shapes = [
     ((1, 12, 10, 64), (1, 12, 64, 10)),
     ((1, 12, 10, 10), (1, 12, 10, 64)),
]


def benchmark_matmul(a_shape, b_shape, dtype=tf.float16, device="cpu", repeat=500):
    # Map PyTorch device string to TensorFlow device string
    if device == "cpu":
        tf_device = "/CPU:0"
    elif device == "cuda":
        tf_device = "/GPU:0"
    else:
        tf_device = device

    with tf.device(tf_device):
        # torch.empty(...).uniform_(0,1) * 2 - 1
        # Creates a tensor with random values in [-1, 1)
        A = tf.random.uniform(a_shape, minval=-1.0, maxval=1.0, dtype=dtype)
        B = tf.random.uniform(b_shape, minval=-1.0, maxval=1.0, dtype=dtype)
        # warm up
        for _ in range(5000):
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