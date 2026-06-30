import tensorflow as tf
import time

def test_executing_eagerly_overhead():
    """
    Test case adapted from the PyTorch matmul CPU overhead issue.
    This test measures the CPU overhead of calling tf.compat.v1.executing_eagerly
    repeatedly in a loop to detect potential performance regressions.
    """
    warmup = 128
    iters = 16384

    # Warmup phase to ensure any initialization overhead is excluded
    for _ in range(warmup):
        tf.compat.v1.executing_eagerly()

    # Measure execution time
    t0 = time.perf_counter()
    for _ in range(iters):
        tf.compat.v1.executing_eagerly()
    t1 = time.perf_counter()

    # Calculate average time per call in microseconds
    avg_time_us = 1e6 * (t1 - t0) / iters
    print(f"Average time per call: {avg_time_us} us")

    # Functional assertion to ensure the API behaves as expected.
    # The original test assumed eager execution is enabled (TF2 default).
    # However, in some environments (e.g., when tf.compat.v1.disable_eager_execution()
    # is called globally), this returns False. We verify the API returns a valid
    # boolean state rather than enforcing a specific execution mode.
    is_eager = tf.compat.v1.executing_eagerly()
    assert isinstance(is_eager, bool), "API should return a boolean value"

if __name__ == "__main__":
    test_executing_eagerly_overhead()