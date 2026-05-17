import torch
import tensorflow as tf
import time

# This test case adapts the CPU overhead benchmarking logic from the PyTorch issue
# to the TensorFlow API `tf.executing_eagerly`.
# The original issue detected a regression in `torch.matmul` by measuring
# the execution time of many iterations. Here we apply the same pattern
# to ensure `tf.executing_eagerly` does not introduce unexpected CPU overhead.

warmup = 128
iters = 16384

# Warmup phase to account for initialization costs
for _ in range(warmup):
    tf.executing_eagerly()

# Benchmarking phase
# Note: Unlike the PyTorch CUDA example, we do not need explicit device synchronization
# because tf.executing_eagerly is a CPU-side context check.
t0 = time.perf_counter()
for _ in range(iters):
    tf.executing_eagerly()
t1 = time.perf_counter()

avg_time_us = 1e6 * (t1 - t0) / iters
print(f"Average time per call (microseconds): {avg_time_us}")

# Assertion to detect significant performance regressions (CPU overhead).
# The threshold is set to a reasonable value (e.g., 1 microsecond) for a simple check.
assert avg_time_us < 1.0, f"Detected measurable CPU overhead: {avg_time_us} us"