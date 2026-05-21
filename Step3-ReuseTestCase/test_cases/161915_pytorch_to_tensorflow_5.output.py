import torch
import tensorflow as tf

# Disable eager execution to use tf.compat.v1.Session and metrics
tf.compat.v1.disable_eager_execution()

def test_false_positives_with_jagged_inputs():
    """
    Adapted test case for tf.compat.v1.metrics.false_positives based on 
    PyTorch NestedTensor share_memory_() bug (Issue 161915).
    
    Original PyTorch Logic:
    1. Create tensors of different sizes (3 and 5).
    2. Combine them into a NestedTensor (jagged layout).
    3. Call a method (share_memory_()) which causes a segfault.
    
    Adapted TensorFlow Logic:
    1. Create labels and predictions mimicking the jagged structure (sizes 3 and 5).
    2. Use tf.RaggedTensor to represent the nested/jagged data.
    3. Call tf.compat.v1.metrics.false_positives to verify behavior with this structure.
    """
    
    # Mimic the input creation: a = torch.randn(3), b = torch.randn(5)
    # We create boolean labels and predictions for two samples of different lengths.
    # Sample 1 (size 3): [False, True, False]
    # Sample 2 (size 5): [True, False, True, False, True]
    
    labels = tf.ragged.constant([
        [0, 1, 0], 
        [1, 0, 1, 0, 1]
    ], dtype=tf.bool)
    
    predictions = tf.ragged.constant([
        [1, 1, 0], 
        [0, 0, 1, 1, 0]
    ], dtype=tf.bool)

    # Call the similar API
    # In PyTorch, the crash happened on share_memory_(). 
    # Here we check if the metric calculation handles the jagged input without crashing.
    try:
        metric, update_op = tf.compat.v1.metrics.false_positives(
            labels=labels,
            predictions=predictions
        )

        with tf.compat.v1.Session() as sess:
            # Initialize local variables required for metrics
            sess.run(tf.compat.v1.local_variables_initializer())
            
            # Run the update operation to calculate the metric
            sess.run(update_op)
            
            # Evaluate the result
            result = sess.run(metric)
            
            print(f"Test Passed. False Positives Result: {result}")
            # Assert result is valid (non-negative number)
            assert result >= 0
            
    except Exception as e:
        print(f"Test Failed with exception: {e}")
        raise

if __name__ == "__main__":
    test_false_positives_with_jagged_inputs()