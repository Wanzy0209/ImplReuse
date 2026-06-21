import sys
import torch
import numpy as np

try:
    import tensorflow as tf
except ImportError as e:
    print(f"Skipping test due to missing dependencies or environment issues: {e}")
    sys.exit(0)

# Test case adapted from PyTorch issue 162504 logic
# Original Issue: torch.utils.checkpoint.checkpoint fails under CUDA graph capture
# Adaptation: Verify tf.keras.backend.set_floatx consistency between Eager and Graph modes

def test_set_floatx_graph_consistency():
    # 1. Setup: Configure global state using the similar API
    # This mirrors the torch.cuda.manual_seed(42) setup in the original issue
    original_floatx = tf.keras.backend.floatx()
    tf.keras.backend.set_floatx('float64')

    # 2. Define the function to be executed
    # Mirrors the fn(x) in the original issue
    def fn(x):
        # Using explicit dtype to ensure the operation respects the precision context
        return x * tf.sigmoid(tf.random.normal((1,), dtype=tf.float64))

    # Initialize device state (warmup)
    fn(tf.ones((1,), dtype=tf.float64))

    # 3. Eager Execution
    tf.random.set_seed(42)
    eager_in = tf.ones((1,), dtype=tf.float64)
    with tf.GradientTape() as tape:
        eager_out = fn(eager_in)
    eager_in_grad = tape.gradient(eager_out, eager_in)

    # 4. Graph Execution (Capture and Replay)
    # tf.function is the TensorFlow equivalent of torch.cuda.graph capture
    tf.random.set_seed(42)
    
    # Capture the graph
    graph_fn = tf.function(fn)
    concrete_fn = graph_fn.get_concrete_function(eager_in)
    
    # Replay the graph
    with tf.GradientTape() as tape:
        graph_out = graph_fn(eager_in)
    graph_in_grad = tape.gradient(graph_out, eager_in)

    # 5. Assertion
    # Check if results match between Eager and Graph execution
    # This mirrors the assert in the original bug report
    assert np.allclose(eager_in_grad.numpy(), graph_in_grad.numpy(), rtol=0.0, atol=0.0), \
        "Mismatch in gradient outputs between Eager and Graph execution"
    
    print("Eager Grad:", eager_in_grad.numpy())
    print("Graph Grad:", graph_in_grad.numpy())

    # Cleanup
    tf.keras.backend.set_floatx(original_floatx)

if __name__ == "__main__":
    test_set_floatx_graph_consistency()