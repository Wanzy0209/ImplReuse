import torch
import tensorflow as tf

def test_redundant_code_side_effect_replacement():
    """
    Test case based on Issue 167064: Redundant global code in the context parallel.
    
    The original issue involved a redundant call to torch.compile that triggered
    a global side effect (torch.distributions.Distribution.set_default_validate_args(False)).
    
    This test replaces:
    1. torch.compile -> tf.keras.ops.relu
    2. torch.distributions.Distribution.set_default_validate_args -> tf.keras.ops.relu
    
    It also leverages the similar API torch.eq (translated to tf.equal) to verify
    the logic, as indicated by the issue-to-api similarity.
    """
    
    # Setup: Create a dummy tensor representing a block mask context
    # In the original issue, this was part of the context parallel attention logic.
    mask_input = tf.constant([[-1.0, 2.0], [0.0, -3.0]])

    # Replacement 1: torch.compile
    # Original: _compiled_create_block_mask = torch.compile(...)
    # Replacement: _compiled_create_block_mask = tf.keras.ops.relu(...)
    # Note: In the original bug, this result was unused (redundant), but we use it here
    # to verify the replacement API behaves as expected.
    _compiled_create_block_mask = tf.keras.ops.relu(mask_input)

    # Replacement 2: torch.distributions.Distribution.set_default_validate_args(False)
    # Original: torch.distributions.Distribution.set_default_validate_args(False)
    # Replacement: tf.keras.ops.relu(False)
    # Note: False is implicitly converted to 0.0 in TensorFlow operations.
    # This line replaces the global side effect call with the specified API.
    validation_replacement = tf.keras.ops.relu(False)

    # Verification using the Similar API pattern (torch.eq -> tf.equal)
    # The torch.eq snippet provided in the issue checks for equality between values.
    # We use tf.equal to verify the result of the ReLU operation.
    expected_mask = tf.constant([[0.0, 2.0], [0.0, 0.0]])
    
    # Check if the mask matches the expected output
    is_equal = tf.equal(_compiled_create_block_mask, expected_mask)
    assert tf.reduce_all(is_equal).numpy(), "Block mask creation failed"

    # Verify the side effect replacement
    # ReLU(False) -> ReLU(0.0) -> 0.0
    assert validation_replacement.numpy() == 0.0, "Validation replacement failed"

if __name__ == "__main__":
    test_redundant_code_side_effect_replacement()
    print("Test passed.")