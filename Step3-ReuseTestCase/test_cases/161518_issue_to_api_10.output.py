import torch
import tensorflow as tf
import time

# This test case adapts the memory throughput benchmarking logic from the PyTorch issue
# (Issue 161518) to the TensorFlow API tf.keras.ops.exp.
#
# Original Logic: Benchmark torch.cat (memory copy) with 512 MiB total IO.
# Adapted Logic: Benchmark tf.keras.ops.exp (elementwise compute) with 512 MiB total IO.
#                (256 MiB Read + 256 MiB Write).

# Target IO size in MiB to match the original script's scale
TOTAL_IO_MIB = 512

# Check for GPU availability to ensure relevance to the original "CUDA" context
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to avoid allocating all memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Running on GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Running on CPU (bandwidth will be significantly lower).")

for dtype in [tf.float32, tf.bfloat16]:
    element_size = tf.dtypes.as_dtype(dtype).size

    # Create input tensor of 256 MiB.
    # For exp: Read (256 MiB) + Write (256 MiB) = 512 MiB Total IO.
    num_elements = (256 * 1024 * 1024) // element_size
    input_tensor = tf.zeros(num_elements, dtype=dtype)

    # Warmup run to initialize kernels and avoid cold start overhead
    _ = tf.keras.ops.exp(input_tensor)

    # Benchmark loop
    # We use time.perf_counter() as a generic equivalent to triton.testing.do_bench
    start_time = time.perf_counter()
    for _ in range(100):
        output = tf.keras.ops.exp(input_tensor)
    end_time = time.perf_counter()

    # Calculate average time in milliseconds
    avg_time_ms = (end_time - start_time) * 1000 / 100

    # Calculate bandwidth in GB/s
    # Formula: (Total IO MiB / Time ms) * 1000 (to get s) / 1024 (to get GB)
    bdwidth = TOTAL_IO_MIB / avg_time_ms * 1000 / 1024

    print("\t".join([str(dtype), f"{avg_time_ms:.2f} ms", f"{bdwidth:.2f} GB/s"]))