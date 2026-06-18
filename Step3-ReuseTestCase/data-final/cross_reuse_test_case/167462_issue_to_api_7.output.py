import torch
import time
import tensorflow as tf

def benchmark_lu_solve():
    """
    Benchmark script for tf.linalg.lu_solve adapted from the torch.topk 
    performance regression report.
    
    The original issue highlights a speed regression in torch.topk when processing
    large tensors (Batch Size=128, Vocab=8M) on float16.
    
    This script adapts the logic to tf.linalg.lu_solve. Since lu_solve operates on
    matrices rather than 1D vocabularies, we adjust the dimension 'N' to be 
    computationally significant but feasible for memory (e.g., 1024x1024).
    """
    
    # Check for GPU availability to match the original CUDA context
    gpus = tf.config.list_physical_devices('GPU')
    device_name = gpus[0].name if gpus else '/CPU:0'
    print(f"Running on device: {device_name}")

    with tf.device(device_name):
        # Adapted dimensions:
        # Original: BS=128, VOCAB=8,000,000
        # Adapted:  BS=128, Matrix Dim=1024 (Large enough to stress compute, fits in memory)
        BS = 128
        N = 1024
        
        # Use float16 to match the original issue's dtype
        # Note: float16 stability for LU solve depends on hardware, but we test the requested dtype.
        try:
            dtype = tf.float16
            # Create a random matrix A and RHS vector b
            # Shape: (Batch, Rows, Cols) for A, (Batch, Rows) for b
            A = tf.random.normal((BS, N, N), dtype=dtype)
            b = tf.random.normal((BS, N), dtype=dtype)
        except tf.errors.InvalidArgumentError:
            print("float16 not fully supported on this device for this op, falling back to float32.")
            dtype = tf.float32
            A = tf.random.normal((BS, N, N), dtype=dtype)
            b = tf.random.normal((BS, N), dtype=dtype)

        # Pre-compute LU factorization. 
        # tf.linalg.lu returns 'lu' (packed lower/upper) and 'p' (permutation).
        # This is the setup cost, separate from the solve latency we want to measure.
        lu, p = tf.linalg.lu(A)

        walltime = []
        iterations = 100
        
        # Warmup (optional, but good for consistency)
        for _ in range(10):
            _ = tf.linalg.lu_solve(lu, p, b)

        for _ in range(iterations):
            start = time.time()
            
            # Operation under test: Solve the linear system
            _ = tf.linalg.lu_solve(lu, p, b)
            
            # In eager mode, operations are synchronous. 
            # If running in graph mode, a session.run() would be needed.
            # We assume eager execution (default in TF 2.x).
            end = time.time()
            
            walltime.append(end - start)

        # Calculate average latency, discarding potential outliers or warmup if not done above
        # The original script discarded the first 10 measurements.
        walltime = walltime[10:]
        avg_latency = sum(walltime) / len(walltime)
        
        print(f"TensorFlow: {tf.__version__}, lu_solve latency: {1000 * avg_latency}ms")

if __name__ == "__main__":
    benchmark_lu_solve()