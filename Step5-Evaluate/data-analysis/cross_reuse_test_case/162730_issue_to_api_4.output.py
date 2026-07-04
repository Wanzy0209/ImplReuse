import tensorflow as tf
import numpy as np

def test_msle_non_contiguous_tensors():
    """
    Test case to verify that tf.keras.metrics.MeanSquaredLogarithmicError
    produces consistent results for contiguous and non-contiguous input tensors.
    
    This test mirrors the logic of the PyTorch bug report (Issue 162730), 
    where numerical inconsistencies were observed between contiguous and 
    non-contiguous tensors in a linear operation.
    """
    
    # 1. Setup test data
    # Using shapes similar to the original bug report context
    batch_size = 1
    seq_len = 3
    features = 768
    
    # Create ground truth and prediction tensors
    y_true = tf.random.uniform((batch_size, seq_len, features))
    y_pred_base = tf.random.uniform((batch_size, seq_len, features))

    # 2. Create non-contiguous tensor
    # Mimicking the 'einops.rearrange' behavior which changes memory layout
    # We transpose to change strides, then reshape to match the original input shape requirements
    # Transpose: (1, 3, 768) -> (768, 3, 1)
    y_pred_transposed = tf.transpose(y_pred_base, perm=[2, 1, 0])
    
    # Reshape: (768, 3, 1) -> (768, 3) -> (1, 3, 768) 
    # This creates a tensor with the same logical shape but non-standard memory layout
    y_pred_noncontig = tf.reshape(y_pred_transposed, (768, 3))
    y_pred_noncontig = tf.reshape(y_pred_noncontig, (batch_size, seq_len, features))

    # 3. Create contiguous version
    # identity() ensures we are working with a standard contiguous copy
    y_pred_contig = tf.identity(y_pred_noncontig)

    # 4. Instantiate the Similar API
    msle_metric = tf.keras.metrics.MeanSquaredLogarithmicError()

    # 5. Compute results with non-contiguous tensor
    msle_metric.reset_states()
    msle_metric.update_state(y_true, y_pred_noncontig)
    result_noncontig = msle_metric.result()

    # 6. Compute results with contiguous tensor
    msle_metric.reset_states()
    msle_metric.update_state(y_true, y_pred_contig)
    result_contig = msle_metric.result()

    # 7. Assertion
    # The original bug showed a mismatch here. We assert that they should match.
    # Fix: Use tf.keras.backend.get_value to handle cases where .numpy() is not available (e.g., graph mode)
    val_noncontig = tf.keras.backend.get_value(result_noncontig)
    val_contig = tf.keras.backend.get_value(result_contig)
    
    is_close = np.allclose(val_noncontig, val_contig, atol=1e-5)
    
    print(f"Non-contiguous result: {val_noncontig}")
    print(f"Contiguous result: {val_contig}")
    print(f"Results match: {is_close}")
    
    assert is_close, (
        f"MeanSquaredLogarithmicError produced inconsistent results between "
        f"contiguous and non-contiguous tensors. "
        f"Diff: {np.abs(val_noncontig - val_contig)}"
    )

if __name__ == "__main__":
    test_msle_non_contiguous_tensors()