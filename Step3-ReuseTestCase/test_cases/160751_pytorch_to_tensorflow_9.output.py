import torch
import tensorflow as tf

def func():
    # Adaptation: Use tf.name_scope to wrap the logic, 
    # acting as the context manager similar to the compile context in the original bug.
    with tf.name_scope("test_scope"):
        # Create a tensor (equivalent to torch.tensor)
        a = tf.constant([1.0, -2.0])
        
        # Perform the check (equivalent to torch.all(a > 0))
        result = tf.reduce_all(a > 0)
        
        # Assert logic
        # We use .numpy() to ensure the tensor is evaluated to a boolean for the assertion.
        # This mimics the synchronization/check point.
        assert result.numpy(), "should throw"
        
        # This print should not run if the assertion is correctly caught
        print("should not run")

def test_fn():
    # Execute the function
    func()

if __name__ == "__main__":
    try:
        test_fn()
        # If we reach here, the behavior is incorrect (similar to the bug where print runs)
        print("FAIL: Code executed past assertion")
    except AssertionError as e:
        # If we catch this, the behavior is correct (assertion caught, print skipped)
        print(f"PASS: Exception caught as expected - {e}")