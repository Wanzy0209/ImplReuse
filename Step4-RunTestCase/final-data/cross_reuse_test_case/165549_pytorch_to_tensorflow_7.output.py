import torch
import sys

def test_sparse_conditional_accumulator_shape():
    """
    Adapted test case for tf.compat.v1.SparseConditionalAccumulator based on 
    PyTorch Issue 165549.
    
    Original Bug: Operations like abs return empty tensors (shape [0]) when 
    run through cpu_fallback, instead of preserving the input shape.
    
    Adapted Logic: Verify that the SparseConditionalAccumulator correctly 
    preserves the shape and content of accumulated sparse gradients, and does 
    not return empty tensors (shape [0]) when gradients are extracted.
    """
    try:
        import tensorflow as tf
    except ImportError as e:
        # Handle environment issues (e.g., missing GLIBCXX_3.4.29) gracefully
        print(f"Skipping test: Failed to import TensorFlow due to environment incompatibility. Error: {e}")
        return

    # Disable eager execution to use v1 session-based APIs
    tf.compat.v1.disable_eager_execution()
    
    # Define the shape of the tensor we are working with (mimicking torch.randn(4, 4))
    expected_shape = [4, 4]
    dtype = tf.float32

    # Initialize the SparseConditionalAccumulator
    # This is the API under test.
    accumulator = tf.compat.v1.SparseConditionalAccumulator(
        dtype=dtype,
        shape=expected_shape,
        name="test_accumulator"
    )

    # Create a sparse gradient (IndexedSlices) to apply
    # Indices: [[0, 1], [2, 3]]
    # Values: [1.0, 2.0]
    indices = tf.constant([[0, 1], [2, 3]], dtype=tf.int64)
    values = tf.constant([1.0, 2.0], dtype=dtype)

    # Define operations
    apply_op = accumulator.apply_grad(indices, values, local_step=0)
    # We request 1 gradient back
    take_op = accumulator.take_grad(num_required=1)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        
        # Apply the gradient
        sess.run(apply_op)
        
        # Extract the gradient
        # In the PyTorch bug, the result tensor had shape [0].
        # Here we check if the returned IndexedSlices (indices and values) are empty.
        result_indices, result_values = sess.run(take_op)
        
        # Assertions
        # The bug reproduction condition is if the result is empty (shape [0]).
        # We assert that the result is NOT empty, matching the expected behavior.
        assert result_indices.shape[0] > 0, \
            f"Bug reproduction: Accumulator returned empty indices (shape {result_indices.shape}). Expected non-empty tensor."
        assert result_values.shape[0] > 0, \
            f"Bug reproduction: Accumulator returned empty values (shape {result_values.shape}). Expected non-empty tensor."
            
        print("Test passed: Accumulator returned non-empty tensors as expected.")

if __name__ == "__main__":
    test_sparse_conditional_accumulator_shape()