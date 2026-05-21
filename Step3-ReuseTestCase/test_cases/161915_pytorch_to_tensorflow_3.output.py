import torch
import tensorflow as tf
import numpy as np

# Disable eager execution to use TF v1 metrics API
tf.compat.v1.disable_eager_execution()

def test_tf_metrics_false_negatives():
    """
    Adapted test case based on PyTorch NestedTensor share_memory_() issue.
    
    Original Logic:
    1. Create random tensors of different sizes (jagged structure).
    2. Construct a specific object (NestedTensor).
    3. Call a state-affecting method (share_memory_).
    4. Observe crash (Segfault).
    
    Adapted Logic for tf.compat.v1.metrics.false_negatives:
    1. Create random data of different sizes (mimicking the jagged input).
    2. Construct the metric object.
    3. Call the update operation (state-affecting).
    4. Verify the API handles the inputs without crashing.
    """
    
    # 1. Generate random data similar to torch.randn
    # Original: a = torch.randn(3), b = torch.randn(5)
    a = np.random.randn(3)
    b = np.random.randn(5)
    
    # Setup placeholders to feed data dynamically
    # The API expects labels and predictions to be castable to bool
    labels_ph = tf.compat.v1.placeholder(tf.bool, shape=[None])
    predictions_ph = tf.compat.v1.placeholder(tf.bool, shape=[None])
    
    # 2. Call the similar API: tf.compat.v1.metrics.false_negatives
    # This creates the metric variables and update ops
    metric, update_op = tf.compat.v1.metrics.false_negatives(
        labels=labels_ph, 
        predictions=predictions_ph
    )
    
    with tf.compat.v1.Session() as sess:
        # Initialize local and global variables for metrics
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        
        # 3. Perform operations (mimicking nt.share_memory_())
        # We feed the data 'a' and 'b' sequentially. 
        # This tests the metric's ability to handle state updates with varying input sizes,
        # analogous to the NestedTensor handling jagged data.
        
        # Update with batch 'a'
        sess.run(update_op, feed_dict={
            labels_ph: a > 0, 
            predictions_ph: a < 0
        })
        
        # Update with batch 'b' (different size)
        sess.run(update_op, feed_dict={
            labels_ph: b > 0, 
            predictions_ph: b < 0
        })
        
        # 4. Verify behavior
        # In the original bug, this step caused a Segmentation fault.
        # Here we assert that the metric returns a valid value without crashing.
        result = sess.run(metric)
        
        assert isinstance(result, (int, float, np.ndarray)), "Metric result should be a number"
        print(f"Test passed. Metric result: {result}")

if __name__ == "__main__":
    test_tf_metrics_false_negatives()