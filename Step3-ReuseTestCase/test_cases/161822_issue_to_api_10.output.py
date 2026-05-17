import tensorflow as tf
import time

# Configuration matching the original issue
warmup = 128
iters = 16384
dim = 512

# Use GPU if available to match the CUDA context of the original bug
device_name = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

with tf.device(device_name):
    # Use bfloat16 if supported, otherwise fallback to float16 to match original intent
    try:
        dtype = tf.bfloat16
    except AttributeError:
        dtype = tf.float16

    a = tf.random.normal((dim, dim), dtype=dtype)

    # Define the operation wrapped in the similar API (tf.control_dependencies)
    # This mimics the scenario where control logic is enforced, potentially adding overhead.
    @tf.function
    def matmul_op(x):
        # Using tf.control_dependencies to explicitly manage execution order,
        # analogous to the synchronization overhead investigated in the PyTorch issue.
        with tf.control_dependencies([]):
            return tf.matmul(x, x)

    # Warmup
    for _ in range(warmup):
        matmul_op(a)

    # Benchmark
    # Note: In TensorFlow, tf.function execution is managed by the runtime,
    # but we measure wall-clock time to detect CPU overhead similar to the original report.
    t0 = time.perf_counter()
    for _ in range(iters):
        matmul_op(a)
    t1 = time.perf_counter()

    print(f"Average time per iteration (microseconds): {1e6 * (t1 - t0) / iters}")