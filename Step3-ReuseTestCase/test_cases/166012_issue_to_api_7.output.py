import torch
import tensorflow as tf
import numpy as np

def test_crossed_column_cache_consistency():
    """
    Test case to verify consistency of tf.feature_column.crossed_column
    behavior across multiple executions (simulating cache hit vs miss scenarios).
    
    This test is inspired by the PyTorch issue where 'tlparse entries on cache 
    hit and not are inconsistent'. Here, we verify that the hashing logic 
    within crossed_column produces consistent results (entries) regardless 
    of whether the underlying execution graph is being traced (miss) or 
    reused (hit) via tf.function.
    """
    
    # 1. Define base categorical columns
    # These act as the 'keys' for the crossed column
    cat_col_a = tf.feature_column.categorical_column_with_vocabulary_list(
        'feature_a', ['x', 'y', 'z'])
    cat_col_b = tf.feature_column.categorical_column_with_vocabulary_list(
        'feature_b', ['p', 'q', 'r'])

    # 2. Define the Similar API: crossed_column
    # This API hashes the cartesian product of inputs into buckets.
    # We need to ensure this hashing is deterministic and consistent.
    crossed_col = tf.feature_column.crossed_column(
        [cat_col_a, cat_col_b], 
        hash_bucket_size=10
    )

    # 3. Prepare for execution
    # Wrap in an indicator column to make it dense and usable in a layer
    indicator_col = tf.feature_column.indicator_column(crossed_col)
    feature_layer = tf.keras.layers.DenseFeatures([indicator_col])

    # 4. Define the execution function with tf.function
    # tf.function introduces a caching mechanism (tracing vs cached execution).
    # This mirrors the torch.compile environment in the original bug.
    @tf.function
    def compute_features(inputs):
        return feature_layer(inputs)

    # 5. Create input data
    # Using specific inputs to test the 'entries' (hashed indices)
    inputs = {
        'feature_a': np.array(['x', 'y']),
        'feature_b': np.array(['p', 'q'])
    }

    # 6. Execution 1: Simulates "Cache Miss" (Tracing)
    # The first call triggers tracing of the tf.function graph.
    output_miss = compute_features(inputs)

    # 7. Execution 2: Simulates "Cache Hit" (Execution)
    # The second call reuses the traced graph.
    output_hit = compute_features(inputs)

    # 8. Assertion
    # The bug report highlights inconsistency in entries/logs.
    # We assert that the actual output (the hashed feature entries) is identical
    # between the trace path and the cached path.
    np.testing.assert_array_equal(
        output_miss.numpy(), 
        output_hit.numpy(),
        err_msg="Crossed column outputs are inconsistent between trace and cached execution."
    )

    print("Test passed: Crossed column behavior is consistent across cache states.")

if __name__ == "__main__":
    test_crossed_column_cache_consistency()