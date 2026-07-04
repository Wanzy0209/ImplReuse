import tensorflow as tf
import time

# Configuration matching the original issue
warmup = 128
iters = 16384
dim = 512

# Use GPU if available to match the CUDA context of the original bug
# Fix: Handle AttributeError for list_physical_devices (TF 1.x or older TF 2.x)
try:
    # Try TF 2.x method (available in TF 2.1+)
    gpu_devices = tf.config.list_physical_devices('GPU')
    has_gpu = len(gpu_devices) > 0
except AttributeError:
    # Fallback for TF 1.x or TF 2.0 where list_physical_devices is missing
    try:
        has_gpu = tf.test.is_gpu_available()
    except:
        has_gpu = False

device_name = '/GPU:0' if has_gpu else '/CPU:0'

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