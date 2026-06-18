import torch
import tensorflow as tf
import time

def benchmark_memory_throughput():
    """
    Benchmarks memory throughput for element-wise operations (tf.math.exp)
    on GPU, similar to the reported issue for torch.Tensor.copy_ and torch.cat.
    """
    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("No GPU found. This benchmark requires CUDA.")
        return

    print(f"Running on GPU: {gpus[0].name}")
    print("Dtype\tTime (ms)\tBandwidth (GiB/s)")

    # Iterate over relevant data types
    # Note: tf.float8_e4m3fn is not as widely supported in standard TF builds as bfloat16/float32,
    # so we stick to the standard types for a runnable test.
    for dtype in [tf.bfloat16, tf.float32]:
        element_size = dtype.size

        # Create 128 MiB input tensor
        # Shape logic adapted from the original PyTorch script:
        # (128 // element_size) * 1024 rows, 1024 cols
        rows = (128 * 1024 * 1024) // (element_size * 1024)
        input_tensor = tf.zeros((rows, 1024), dtype=dtype)

        # Warmup run to allocate memory and initialize kernels
        _ = tf.math.exp(input_tensor)
        
        # Synchronize to ensure accurate timing
        # (In eager mode, .numpy() forces synchronization)
        _ = input_tensor.numpy()

        # Benchmark loop
        reps = 100
        start_time = time.time()
        
        for _ in range(reps):
            output = tf.math.exp(input_tensor)
            
        # Force execution/synchronization
        _ = output.numpy()
        
        end_time = time.time()

        # Calculate average time in milliseconds
        avg_time_ms = (end_time - start_time) / reps * 1000

        # Calculate Bandwidth
        # Operation: Read 128 MiB (input) + Write 128 MiB (output) = 256 MiB total IO
        io_size_mib = 256
        bdwidth = io_size_mib / avg_time_ms * 1000 / 1000  # GiB/s

        print("\t".join([str(dtype.name), f"{avg_time_ms:.2f}", f"{bdwidth:.2f}"]))

if __name__ == "__main__":
    benchmark_memory_throughput()