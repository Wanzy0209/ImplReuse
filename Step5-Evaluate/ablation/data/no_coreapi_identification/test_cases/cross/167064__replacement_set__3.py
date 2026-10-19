import torch
import tensorflow as tf

def test_redundant_global_code_with_gather():
    """
    Test case reflecting the logic of Issue 167064 (Redundant global code in context parallel).
    The original issue involved torch.compile creating unused code that triggered a global side effect
    (torch.distributions.Distribution.set_default_validate_args(False)).
    
    This test replaces:
    - torch.compile with tf.gather
    - torch.distributions.Distribution.set_default_validate_args with tf.gather
    
    It preserves the logic of performing a redundant operation followed by a state check,
    leveraging the similarity with torch.eq for verification.
    """
    # Setup data simulating the context parallel attention mask parameters
    params = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
    indices = tf.constant([0, 2, 4])

    # Step 1: Simulate the redundant global code creation
    # Original API: torch.compile (called to create _compiled_create_block_mask)
    # Replacement API: tf.gather
    # In the original bug, this line was executed but the result was never used,
    # yet it triggered global side effects.
    _ = tf.gather(params, indices)

    # Step 2: Simulate the side effect / validation check
    # Original API: torch.distributions.Distribution.set_default_validate_args(False)
    # Replacement API: tf.gather
    # We perform the gather operation to verify the state/data.
    # This replaces the 'set_default_validate_args' call which was the problematic side effect.
    result = tf.gather(params, indices)

    # Step 3: Verification
    # Similar API: torch.eq (checks equality)
    # We verify that the result matches the expected output using equality checks.
    expected = tf.constant([1.0, 3.0, 5.0])
    
    # Using tf.equal to mimic the logic of torch.eq
    is_equal = tf.equal(result, expected)
    
    # Assert that all elements are equal
    assert tf.reduce_all(is_equal).numpy(), "Test failed: tf.gather results do not match expected values"

if __name__ == "__main__":
    test_redundant_global_code_with_gather()