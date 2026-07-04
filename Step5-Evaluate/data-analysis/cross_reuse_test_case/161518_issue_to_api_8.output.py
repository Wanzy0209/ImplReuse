import tensorflow as tf
import time

def test_elementwise_exp_throughput():
    """
    Test case to benchmark memory throughput for tf.experimental.numpy.exp.
    Adapted from PyTorch Issue 161518 which benchmarks memory throughput 
    for elementwise operations (copy_/cat).
    """
    # Check for GPU availability
    # Use try-except to handle different TensorFlow versions (TF 1.x vs 2.x)
    try:
        gpus = tf.config.list_physical_devices('GPU')
    except AttributeError:
        # Fallback for older TensorFlow versions (e.g., 1.x)
        gpus = tf.test.is_gpu_available()

    if not gpus:
        print("Test skipped: No GPU available.")
        return

    print("Benchmarking tf.experimental.numpy.exp memory throughput")

    # Original issue used float8_e4m3fn, bfloat16, float32.
    # We use float32 and bfloat16 for compatibility.
    dtypes = [tf.float32, tf.bfloat16]

    for dtype in dtypes:
        element_size = dtype.size

        # Create 128 MiB input tensor (similar to input1 in original issue)
        # 128 MiB = 128 * 1024 * 1024 bytes
        num_elements = (128 * 1024 * 1024) // element_size
        
        # Reshape to (X, 1024) to match original script's structure
        dim1 = 1024
        dim2 = num_elements // dim1

        with tf.device('/GPU:0'):
            input_tensor = tf.zeros((dim1, dim2), dtype=dtype)

            # Warmup run
            _ = tf.experimental.numpy.exp(input_tensor)

            # Benchmark
            # Original used rep=100
            rep = 100
            start_time = time.time()
            for _ in range(rep):
                # Perform element-wise operation using the similar API
                output = tf.experimental.numpy.exp(input_tensor)
            
            # Force synchronization to get accurate GPU time
            _ = output.numpy()
            end_time = time.time()

            avg_time_ms = (end_time - start_time) / rep * 1000

            # Calculate Bandwidth
            # For exp: Read 128 MiB + Write 128 MiB = 256 MiB total IO
            # Original formula: Total_MiB / Time_ms * 1000 / 1000 / 1000 = TiB/s
            total_io_mib = 256
            bdwidth = total_io_mib / avg_time_ms * 1000 / 1000 / 1000

            print("\t".join([str(dtype), f"{avg_time_ms:.2f} ms", f"{bdwidth:.2f} TiB/s"]))

            # Basic correctness assertion
            assert output.shape == input_tensor.shape
            assert output.dtype == input_tensor.dtype

if __name__ == "__main__":
    test_elementwise_exp_throughput()