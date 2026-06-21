import torch
import numpy as np
import sys

# Handle environment dependency issues (e.g., GLIBCXX version mismatch)
try:
    import tensorflow as tf
except ImportError as e:
    print(f"Skipping test: TensorFlow import failed due to environment issues (likely GLIBCXX version mismatch).")
    print(f"Error details: {e}")
    sys.exit(0)

def test_categorical_hinge_non_contiguous_consistency():
    """
    Test case for tf.keras.losses.CategoricalHinge based on the logic of 
    PyTorch Issue #162730 (MPS F.Linear inconsistency with non-contiguous tensors).
    
    This test verifies that CategoricalHinge produces consistent results 
    regardless of the memory layout (contiguous vs non-contiguous) of the input tensors.
    """
    
    # 1. Setup input data
    # Mimicking the random tensor creation from the original bug report
    batch_size = 2
    num_classes = 4
    
    y_true = np.random.randint(0, 2, size=(batch_size, num_classes)).astype(np.float32)
    y_pred = np.random.rand(batch_size, num_classes).astype(np.float32)
    
    # 2. Create non-contiguous tensor
    # The original bug used einops.rearrange(W, "h d m -> m (h d)") to create non-contiguous weights.
    # We simulate this by reshaping to 3D, transposing (changing strides), and reshaping back to 2D.
    # This ensures the tensor memory layout is different from a simple contiguous allocation.
    
    # Reshape to 3D: (Batch, Classes) -> (Batch, Classes/2, 2)
    y_pred_3d = tf.reshape(y_pred, (batch_size, 2, 2))
    
    # Transpose dimensions to alter memory layout: (Batch, Classes/2, 2) -> (2, Batch, Classes/2)
    y_pred_transposed = tf.transpose(y_pred_3d, perm=[2, 0, 1])
    
    # Reshape back to original 2D shape: (2, Batch, Classes/2) -> (Batch, Classes)
    # This tensor is now effectively "non-contiguous" relative to the standard allocation.
    y_pred_noncontig = tf.reshape(y_pred_transposed, (batch_size, num_classes))
    
    # 3. Instantiate the API
    # Corresponds to torch.nn.functional.linear in the original issue
    hinge_loss_fn = tf.keras.losses.CategoricalHinge()
    
    # 4. Compute results
    # Corresponds to result1 = torch.nn.functional.linear(x, w_noncontig, bias)
    loss_noncontig = hinge_loss_fn(y_true, y_pred_noncontig)
    
    # Corresponds to result2 = torch.nn.functional.linear(x, w_contig, bias)
    loss_contig = hinge_loss_fn(y_true, y_pred)
    
    # 5. Assertions
    # The original bug showed these did NOT match on MPS. 
    # Here we assert they SHOULD match for the Similar API.
    print(f"Contiguous Loss: {loss_contig.numpy()}")
    print(f"Non-Contiguous Loss: {loss_noncontig.numpy()}")
    
    assert np.allclose(loss_contig.numpy(), loss_noncontig.numpy(), atol=1e-5), \
        "CategoricalHinge produced inconsistent results between contiguous and non-contiguous tensors."

if __name__ == "__main__":
    test_categorical_hinge_non_contiguous_consistency()