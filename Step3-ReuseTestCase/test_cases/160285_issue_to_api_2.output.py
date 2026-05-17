import tensorflow as tf
import tf.experimental.dtensor as dtensor
import os
import tempfile
import numpy as np

def test_sharded_save_no_data_duplication():
    """
    Test case for tf.experimental.dtensor.sharded_save inspired by 
    Issue #160285 (Wrong-size gradients in Expert Parallel MoE).
    
    The original bug describes a scenario where distributed tensor operations
    (specifically gradient accumulation in Expert Parallelism) resulted in 
    values being doubled (2x the expected size).
    
    This test adapts that logic to tf.experimental.dtensor.sharded_save. 
    It verifies that saving a sharded tensor does not result in data 
    duplication or incorrect aggregation (e.g., values being doubled) 
    when the checkpoint is restored.
    """
    
    # 1. Setup the distributed mesh
    # We use a 1D mesh with 2 devices to simulate the parallel environment 
    # similar to the 'nproc_per_node=2' in the original bug report.
    # Using CPU devices to ensure the test runs in most environments.
    devices = ["CPU:0", "CPU:1"]
    mesh = dtensor.create_mesh([("batch", 2)], devices=devices)

    # 2. Create a DTensor with known values
    # We initialize a tensor with 1.0s. If a bug similar to the MoE gradient 
    # issue exists (where gradients were doubled), the saved data might 
    # incorrectly aggregate to 2.0s.
    shape = [2, 4]
    initial_value = tf.ones(shape, dtype=tf.float32)
    
    # Shard the tensor along the first dimension ('batch')
    # Each device will hold a slice of the data.
    layout = dtensor.Layout([dtensor.SHARDED, dtensor.UNSHARDED], mesh)
    sharded_tensor = dtensor.DTensor(initial_value, layout=layout, mesh=mesh)

    # 3. Save the tensor using the API under test
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "ckpt")
        
        # Call the similar API: tf.experimental.dtensor.sharded_save
        dtensor.sharded_save(
            mesh=mesh,
            file_prefix=prefix,
            tensor_names=["test_tensor"],
            shape_and_slices=[""], # No slicing, full tensor save
            tensors=[sharded_tensor]
        )

        # 4. Verify the saved data integrity
        # Load the checkpoint back to check if the values were preserved correctly.
        reader = tf.train.load_checkpoint(prefix)
        restored_tensor = reader.get_tensor("test_tensor")

        # Expected value is the original 1.0s.
        # In the original bug, gradients were 2x size. Here we check for 2x values.
        expected_value = np.ones(shape, dtype=np.float32)
        
        # Assert that the values are NOT doubled (or otherwise incorrect)
        is_correct = np.allclose(restored_tensor, expected_value)
        
        if not is_correct:
            print(f"FAILURE: Data integrity check failed.")
            print(f"Expected (all 1.0s):\n{expected_value}")
            print(f"Got (potential duplication detected):\n{restored_tensor}")
            raise AssertionError(
                "tf.experimental.dtensor.sharded_save produced incorrect output. "
                "Values appear to be duplicated or scaled, similar to the MoE gradient bug."
            )
        else:
            print("SUCCESS: Data integrity preserved. No duplication detected.")

if __name__ == "__main__":
    test_sharded_save_no_data_duplication()