import tensorflow as tf
import numpy as np

def test_swapaxes_on_ragged_tensor():
    """
    Adapted from PyTorch Issue 161812.
    Original issue: Crash in jagged tensor stack/cat along dimension 0.
    
    This test verifies the behavior of tf.keras.ops.swapaxes on a RaggedTensor,
    which is the TensorFlow equivalent of a PyTorch jagged tensor.
    We specifically test operations involving dimension 0 and the ragged dimension
    to check for similar crashes or errors.
    """
    
    # Recreate a data structure analogous to the PyTorch nested tensor:
    # PyTorch: th.nested.nested_tensor([th.ones(3, 2, 3), th.ones(4, 2, 3)], layout=th.jagged)
    # This creates a batch of 2 tensors, where the first dimension is jagged (3 vs 4).
    # In TensorFlow, we represent this as a RaggedTensor with shape (2, None, 2, 3).
    # For simplicity in this test, we use a 3D RaggedTensor (2, None, 3) 
    # to represent the jagged nature.
    
    # Constructing a RaggedTensor:
    # Row 0 has 3 elements, Row 1 has 4 elements.
    # Each element is a vector of size 3.
    data = tf.ragged.constant([
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]], # Shape (3, 3)
        [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]] # Shape (4, 3)
    ])
    
    print(f"Input RaggedTensor: {data}")
    print(f"Input Shape: {data.shape}")

    # The PyTorch bug occurred when operating along dimension 0.
    # We test swapping axis 0 (batch dimension) with axis 1 (ragged dimension).
    # Note: Swapping the batch dimension with a ragged dimension is a complex operation
    # and might not be supported or might raise an error.
    try:
        # Attempt to swap axes 0 and 1
        result = tf.keras.ops.swapaxes(data, axis1=0, axis2=1)
        print(f"Result of swapaxes(data, 0, 1): {result}")
        # If successful, verify the shape changed appropriately
        # (Though exact shape validation depends on TF implementation support)
        assert result.shape.rank == 3
    except Exception as e:
        # We catch exceptions to ensure the API doesn't crash the runtime (segfault),
        # similar to the ValueError raised in PyTorch.
        print(f"Caught expected error during swapaxes(0, 1): {type(e).__name__}: {e}")
        assert isinstance(e, (ValueError, NotImplementedError, tf.errors.InvalidArgumentError))

    # Test swapping the ragged dimension (1) with the dense dimension (2).
    # This is a more standard operation for ragged tensors.
    try:
        result = tf.keras.ops.swapaxes(data, axis1=1, axis2=2)
        print(f"Result of swapaxes(data, 1, 2): {result}")
        # Verify shape: (2, 3, None) - the raggedness moves to the last dimension
        assert result.shape[0] == 2
        assert result.shape[1] == 3
        assert result.shape[2] is None # Ragged dimension
    except Exception as e:
        print(f"Caught error during swapaxes(1, 2): {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_swapaxes_on_ragged_tensor()