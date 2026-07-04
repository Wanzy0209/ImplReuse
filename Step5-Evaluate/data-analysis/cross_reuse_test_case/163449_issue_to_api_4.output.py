import torch
import tensorflow as tf
import numpy as np

def test_dropout_eager_graph_consistency():
    """
    Test case for tf.compat.v1.nn.dropout inspired by the PyTorch eager/compile divergence bug.
    
    The original issue (Issue 163449) highlights a numerical divergence between eager 
    and compiled modes in PyTorch when using bfloat16 and specific math operations.
    
    This test adapts that logic to the similar API (tf.compat.v1.nn.dropout) by checking
    for consistency between TensorFlow's eager execution and graph execution (tf.function)
    using bfloat16 precision, mirroring the conditions of the original bug.
    """
    
    # Enable eager execution explicitly (default in TF2, but good for clarity)
    tf.config.run_functions_eagerly(False)

    # Setup inputs similar to the original bug's sensitivity (bfloat16, specific shapes)
    # Original bug used (5, 1024) tensors
    x = tf.random.uniform((5, 1024), minval=-1.0, maxval=1.0, dtype=tf.bfloat16, seed=42)
    rate = 0.5
    seed = 1234  # Seed is required by the similar API implementation to ensure reproducibility

    # 1. Eager Execution
    # This corresponds to the 'out_eager' in the original bug report
    y_eager = tf.compat.v1.nn.dropout(x, rate=rate, seed=seed)

    # 2. Graph Execution (tf.function)
    # This corresponds to the 'torch.compile' execution in the original bug report
    @tf.function
    def graph_dropout(x_in):
        return tf.compat.v1.nn.dropout(x_in, rate=rate, seed=seed)

    y_graph = graph_dropout(x)

    # 3. Compare Outputs
    # We cast to float32 for comparison to avoid bfloat16 precision issues in the check itself
    diff = tf.reduce_max(tf.abs(tf.cast(y_eager, tf.float32) - tf.cast(y_graph, tf.float32)))
    
    # The original bug reported > 5% difference. Here we assert that the difference is negligible
    # (effectively zero, since dropout with a fixed seed should be deterministic).
    tolerance = 1e-5
    
    print(f"Max difference between Eager and Graph: {diff.numpy()}")
    
    if diff > tolerance:
        print(f" Divergence detected! Max diff: {diff.numpy()}")
        print("Eager output sample:", y_eager.numpy().flatten()[:5])
        print("Graph output sample:", y_graph.numpy().flatten()[:5])
        raise AssertionError(f"Eager and Graph outputs differ by more than {tolerance}")
    else:
        print(" Test Passed: No significant divergence between Eager and Graph modes.")

if __name__ == "__main__":
    test_dropout_eager_graph_consistency()