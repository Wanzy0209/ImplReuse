import tensorflow as tf
import time

# Fix: Enable eager execution to allow .numpy() access and accurate timing
# This is required because the error indicates the environment is running in Graph mode
# where .numpy() is not available on Tensors.
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()

# Benchmarking memory throughput for tf.keras.backend.arange
# Adapted from the PyTorch copy_/cat benchmark logic to test elementwise generation throughput

target_mib = 128  # 128 MiB data target

# Using dtypes supported by tf.keras.backend.arange
dtypes = [tf.float16, tf.float32, tf.int32]

print("DType\t\tTime (ms)\tBandwidth (TiB/s)")

for dtype in dtypes:
    # Fix: Use as_numpy_dtype.itemsize to get element size robustly
    # dtype.size is not a standard attribute in TensorFlow DType objects
    element_size = dtype.as_numpy_dtype.itemsize
    
    # Calculate number of elements to reach target size
    num_elements = (target_mib * 1024 * 1024) // element_size

    # Warmup (eager execution)
    _ = tf.keras.backend.arange(0, num_elements, 1, dtype=dtype)

    # Fix: Define tf.function outside the loop to avoid re-tracing overhead
    # which would skew the benchmark results.
    @tf.function
    def bench():
        return tf.keras.backend.arange(0, num_elements, 1, dtype=dtype)
    
    # Warmup the graph function (trace it)
    _ = bench().numpy()

    # Benchmark loop
    start = time.time()
    for _ in range(100):
        t = bench()
        # Force synchronization to ensure accurate timing
        # This works now because eager execution is enabled
        _ = t.numpy()
        
    end = time.time()
    
    elapsed_ms = (end - start) * 1000 / 100
    
    # Calculate Bandwidth in TiB/s
    # Formula: (Data_MiB / Time_ms) * 1000 (ms/s) / 1024 / 1024 (MiB to TiB)
    bdwidth = (target_mib / elapsed_ms) * 1000 / 1024 / 1024

    print(f"{dtype.name}\t\t{elapsed_ms:.2f}\t\t{bdwidth:.2f}")