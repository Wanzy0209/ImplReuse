import torch
import tensorflow as tf
import tempfile
import shutil
import os

def test_mirrored_serialization_multiple_runs():
    """
    Test case adapted from PyTorch AOT Precompile serialization issue (ID: 165447).
    
    Original Bug Logic:
    1. Create a model/compiled function.
    2. Save the compiled function to disk.
    3. Reset compiler state.
    4. Load the function from disk.
    5. Verify execution and correctness.
    
    Adapted Logic for tf.types.experimental.distributed.Mirrored:
    1. Create a MirroredVariable within a strategy scope.
    2. Save the variable state using a Checkpoint.
    3. Simulate a reset by creating a new variable instance.
    4. Load the state into the new variable.
    5. Verify the variable is still a Mirrored type and values match.
    6. Repeat to test stability over multiple runs (as per bug title).
    """
    
    # Initialize strategy to create Mirrored variables
    strategy = tf.distribute.MirroredStrategy()

    # Run the serialization cycle multiple times to catch potential state leakage
    # or corruption, similar to the "running multiple times" aspect of the bug.
    for run_id in range(2):
        print(f"Run {run_id + 1}")
        
        # 1. Create and Initialize State
        with strategy.scope():
            # Creating a variable here results in a MirroredVariable
            original_var = tf.Variable([run_id * 10.0], name="mirrored_test_var")
            expected_value = original_var.read_value()

        # 2. Serialize (Save)
        checkpoint_dir = tempfile.mkdtemp()
        checkpoint = tf.train.Checkpoint(var=original_var)
        save_path = checkpoint.save(checkpoint_dir)

        # 3. Reset / Load Context
        # We create a new variable to simulate loading into a fresh environment,
        # analogous to torch._dynamo.reset() and loading the compiled function.
        with strategy.scope():
            loaded_var = tf.Variable([0.0], name="mirrored_test_var")

        # 4. Deserialize (Load)
        new_checkpoint = tf.train.Checkpoint(var=loaded_var)
        status = new_checkpoint.restore(save_path)
        status.assert_existing_objects_matched()

        # 5. Verification
        # Check that the loaded object is indeed of the Mirrored type
        assert isinstance(loaded_var, tf.types.experimental.distributed.Mirrored), \
            f"Expected loaded variable to be Mirrored, got {type(loaded_var)}"

        # Check that the values match the original state
        assert tf.reduce_all(tf.equal(loaded_var.read_value(), expected_value)), \
            f"Value mismatch after load. Expected {expected_value}, got {loaded_var.read_value()}"

        # Check that the variable is functional (mimicking the forward pass assertion)
        # We perform a simple operation to ensure the internal state is valid.
        result = loaded_var + 1
        assert tf.reduce_all(tf.equal(result, expected_value + 1)), \
            "Computation on loaded variable failed."

        # Cleanup
        shutil.rmtree(checkpoint_dir)

    print("Test passed: Mirrored variable serialization is stable across multiple runs.")

if __name__ == "__main__":
    test_mirrored_serialization_multiple_runs()