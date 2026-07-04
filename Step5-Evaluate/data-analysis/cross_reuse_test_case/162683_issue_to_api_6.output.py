import torch
import time
import numpy as np
import sys

# Handle environment incompatibility (e.g., GLIBCXX version mismatch)
try:
    import tensorflow as tf
except ImportError as e:
    print("Skipping test: TensorFlow cannot be imported due to environment incompatibility.")
    print(f"ImportError: {e}")
    sys.exit(0)

def benchmark_crossed_column(repeat=500):
    """
    Benchmarks the performance of tf.feature_column.crossed_column
    using a similar loop structure to the torch.matmul regression report.
    """
    # Define categorical columns to be crossed
    # Mimicking the dimensionality of the original issue (e.g., 10 and 64 buckets)
    col_a = tf.feature_column.categorical_column_with_identity('feature_a', num_buckets=10)
    col_b = tf.feature_column.categorical_column_with_identity('feature_b', num_buckets=64)
    
    # Create the crossed column
    crossed_col = tf.feature_column.crossed_column([col_a, col_b], hash_bucket_size=1000)
    
    # DenseFeatures is required to execute the feature column logic
    crossed_layer = tf.keras.layers.DenseFeatures([crossed_col])

    # Create input data
    # Using a batch size of 1 to match the (1, ...) shapes in the original issue
    inputs = {
        'feature_a': tf.constant(np.random.randint(0, 10, size=(1, 1)), dtype=tf.int64),
        'feature_b': tf.constant(np.random.randint(0, 64, size=(1, 1)), dtype=tf.int64),
    }

    # Warm up
    # Running 5000 iterations to stabilize performance, matching the original logic
    for _ in range(5000):
        _ = crossed_layer(inputs)

    # Run benchmark
    times = []
    for i in range(repeat):
        start = time.time()
        _ = crossed_layer(inputs)
        end = time.time()
        
        # Discard the first 100 measurements to account for initialization overhead
        if i > 100:
            times.append(round((end - start) * 1000 * 1000))
            
    times.sort()
    print("Microseconds per run (sorted):", times[:10], "...") # Print first 10 for brevity
    
    if times:
        avg_time_us = sum(times) / len(times)
        return avg_time_us
    else:
        return 0.0

if __name__ == "__main__":
    print("Benchmarking tf.feature_column.crossed_column...")
    avg_time = benchmark_crossed_column()
    print(f"Average time: {avg_time:.3f} us")