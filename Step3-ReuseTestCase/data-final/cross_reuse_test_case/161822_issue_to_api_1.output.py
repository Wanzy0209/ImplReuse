import tensorflow as tf
import time

# This test case is adapted from the PyTorch matmul CPU overhead issue.
# It benchmarks the tf.feature_column.crossed_column operation to detect
# similar performance regressions (CPU overhead) in tight loops.

def test_crossed_column_cpu_overhead():
    warmup = 128
    iters = 16384

    # Setup feature columns
    # We use identity columns as inputs for the crossed column
    col_a = tf.feature_column.categorical_column_with_identity('a', num_buckets=100)
    col_b = tf.feature_column.categorical_column_with_identity('b', num_buckets=100)
    
    # API Under Test: tf.feature_column.crossed_column
    crossed_col = tf.feature_column.crossed_column([col_a, col_b], hash_bucket_size=1000)

    # To execute the feature column logic, we use a DenseFeatures layer
    layer = tf.keras.layers.DenseFeatures([crossed_col])

    # Create input tensors
    # Using a small batch size to emphasize overhead
    inputs = {
        'a': tf.constant([[1], [2], [3]] * 10),
        'b': tf.constant([[4], [5], [6]] * 10)
    }

    # Warmup runs to ensure any initialization/caching is done
    for _ in range(warmup):
        _ = layer(inputs)

    # Benchmarking loop
    # We measure the time taken to execute the operation repeatedly
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = layer(inputs)
    t1 = time.perf_counter()

    avg_time_us = 1e6 * (t1 - t0) / iters
    print(f"Average time per iteration (us): {avg_time_us}")
    
    # In a real regression test, you might assert against a known threshold
    # assert avg_time_us < 10.0, "CPU overhead for crossed_column has increased"

if __name__ == "__main__":
    test_crossed_column_cpu_overhead()