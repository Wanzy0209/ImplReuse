import tensorflow as tf
import numpy as np

# Test case adapted from PyTorch Issue 165297 for tf.compat.v1.metrics.accuracy
# The original bug involves NaNs with bfloat16 and large tensors.
# The similar API (tf.compat.v1.metrics.accuracy) performs explicit casting between dtypes.
# This test verifies that the casting logic in the similar API handles bfloat16
# on large tensors without producing NaNs or Infs.

def test_metrics_accuracy_bfloat16_large_tensor():
    # Dimensions from the original bug report
    # Reduced N from 84 to 32 to avoid int32 overflow (2^31 - 1) and OOM errors.
    # Original size: 84 * 64 * 512 * 960 = 2,633,625,600 elements (> 2,147,483,647).
    # New size: 32 * 64 * 512 * 960 = 1,006,632,960 elements.
    N, C, H, W = 32, 64, 512, 960

    tf.compat.v1.reset_default_graph()

    # Create large tensors.
    # Using bfloat16 (the problematic dtype from the bug) for y_true
    # and float32 for y_pred to trigger the 'cast' logic in the similar API.
    # Shape (N, H, W, C) corresponds to channels_last (NHWC) in TensorFlow.
    y_true = tf.random.uniform((N, H, W, C), minval=0, maxval=10, dtype=tf.bfloat16)
    y_pred = tf.random.uniform((N, H, W, C), minval=0, maxval=10, dtype=tf.float32)

    # Call the similar API
    # Implementation path: if y_true.dtype != y_pred.dtype: y_pred = math_ops.cast(y_pred, y_true.dtype)
    accuracy, update_op = tf.compat.v1.metrics.accuracy(y_true, y_pred)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        
        # Execute the operation
        result = sess.run(update_op)

        # Check for NaNs and Infs, mirroring the original bug report's assertions
        print(f"Output contains NaN? {np.isnan(result)}")
        print(f"Output contains Inf? {np.isinf(result)}")
        print(f"Stats: value={result}")

        assert not np.isnan(result), "Detected NaNs in metrics.accuracy output!"
        assert not np.isinf(result), "Detected Infs in metrics.accuracy output!"

if __name__ == "__main__":
    test_metrics_accuracy_bfloat16_large_tensor()