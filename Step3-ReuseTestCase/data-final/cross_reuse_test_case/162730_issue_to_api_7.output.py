import torch
import tensorflow as tf
import numpy as np

def test_mape_non_contiguous_consistency():
    """
    Test case to verify that tf.keras.losses.MeanAbsolutePercentageError
    produces consistent results for contiguous and non-contiguous tensors.
    
    This test is adapted from the logic of PyTorch Issue #162730, where
    torch.nn.functional.linear produced inconsistent results on MPS
    depending on the memory layout of the weight tensor.
    """
    
    # 1. Create base tensors
    # Using float32 and avoiding zeros to prevent division by zero in MAPE
    y_true = tf.random.uniform((10, 20), minval=1.0, maxval=10.0, dtype=tf.float32)
    y_pred = tf.random.uniform((10, 20), minval=1.0, maxval=10.0, dtype=tf.float32)

    # 2. Create non-contiguous tensors via slicing (strided slice)
    # This mimics the 'einops.rearrange' or non-contiguous creation in the bug report.
    # Slicing with a step > 1 usually results in a non-contiguous memory layout.
    y_true_noncontig = y_true[:, ::2]
    y_pred_noncontig = y_pred[:, ::2]

    # 3. Create contiguous versions by copying the non-contiguous tensors
    # This ensures the data is the same, but the memory is contiguous.
    y_true_contig = tf.identity(y_true_noncontig)
    y_pred_contig = tf.identity(y_pred_noncontig)

    # 4. Initialize the loss function
    mape = tf.keras.losses.MeanAbsolutePercentageError()

    # 5. Calculate loss for non-contiguous inputs
    loss_noncontig = mape(y_true_noncontig, y_pred_noncontig)

    # 6. Calculate loss for contiguous inputs
    loss_contig = mape(y_true_contig, y_pred_contig)

    # 7. Assert that the results are identical
    # This mirrors the logic in the bug report: checking if results match
    # between contiguous and non-contiguous inputs.
    # We use numpy for the assertion to handle scalar comparison easily.
    diff = np.abs(loss_noncontig.numpy() - loss_contig.numpy())
    
    print(f"Loss (Non-Contiguous): {loss_noncontig.numpy()}")
    print(f"Loss (Contiguous):     {loss_contig.numpy()}")
    print(f"Difference:            {diff}")

    assert diff < 1e-5, (
        f"MeanAbsolutePercentageError produced inconsistent results "
        f"between contiguous and non-contiguous tensors. "
        f"Diff: {diff}"
    )

if __name__ == "__main__":
    test_mape_non_contiguous_consistency()
    print("Test passed: MAPE is consistent for contiguous and non-contiguous tensors.")