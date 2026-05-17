import torch
import tensorflow as tf

def test_rot90_negative_k():
    """
    Test case for tf.image.rot90 based on the logic of handling negative parameters,
    similar to the negative padding issue in torch.ops.aten.constant_pad_nd (Issue 161014).
    
    The original bug highlights inconsistent behavior when negative values are passed
    to dimension-manipulating arguments. This test verifies that tf.image.rot90
    handles negative 'k' arguments (rotation steps) consistently.
    """
    # Create a simple 3D tensor [height, width, channels]
    # Shape: (2, 2, 1)
    image = tf.constant([[[1], [2]],
                         [[3], [4]]])

    # Test Case 1: Negative k value
    # In the original bug, negative padding values were used to shrink dimensions.
    # Here, we use a negative k value to rotate in the opposite direction.
    # k=-1 should be equivalent to k=3 (270 degrees counter-clockwise)
    result_neg = tf.image.rot90(image, k=-1)
    result_pos = tf.image.rot90(image, k=3)

    # Assert that the operation completes without error and produces the expected result.
    # This checks for consistency between negative and positive parameter handling.
    assert tf.reduce_all(tf.equal(result_neg, result_pos)).numpy(), \
        "tf.image.rot90 failed to handle negative k correctly compared to positive equivalent."

    # Test Case 2: Verify shape consistency
    # The PyTorch bug resulted in unexpected shapes (e.g., [5, 0]). 
    # rot90 should preserve the input shape regardless of k.
    assert result_neg.shape == image.shape, \
        f"Shape changed unexpectedly: {result_neg.shape} vs {image.shape}"

    print("Test passed: Negative k handling in tf.image.rot90 is consistent.")

if __name__ == "__main__":
    test_rot90_negative_k()