import torch
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_v2 as fc

# Test case adapted from PyTorch issue 163798
# Original Issue: tolist() is graphed (lowered to getitem/item) while item() breaks the graph.
# This test verifies the graphing behavior of the similar API (crossed_column) inside tf.function.
# It checks if the operation involving list-like inputs (keys) is handled consistently.

def test_crossed_column_graph_behavior():
    # Setup feature columns analogous to the input tensor 'a'
    col_a = fc.categorical_column_with_identity('a', num_buckets=3)
    col_b = fc.categorical_column_with_identity('b', num_buckets=3)

    # The similar API: crossed_column
    # It takes a list of keys, mirroring the list output of tolist() in the original issue
    crossed_col = fc.crossed_column(keys=[col_a, col_b], hash_bucket_size=10)

    # Wrap in tf.function (analogous to torch.compile)
    @tf.function
    def func(features):
        # In PyTorch: u0, u1 = a.tolist()
        # Here we use crossed_column which combines features based on the list of keys
        indicator = tf.feature_column.indicator_column(crossed_col)
        
        # In PyTorch: return a*u0*u1
        # Here we apply the transformation to get a dense tensor result
        dense_layer = tf.keras.layers.DenseFeatures([indicator])
        return dense_layer(features)

    # Input data
    features = {
        'a': tf.constant([0, 1, 2]),
        'b': tf.constant([1, 2, 0])
    }

    # Execute the function
    result = func(features)

    # Assertions to verify correct graphing and output
    # The graph should execute without breaking, similar to the tolist() case
    assert result.shape == (3, 10), f"Expected shape (3, 10), got {result.shape}"
    assert isinstance(result, tf.Tensor), "Expected a Tensor result from graph execution"
    
    print("Test passed: crossed_column graphs correctly in tf.function.")

if __name__ == "__main__":
    test_crossed_column_graph_behavior()