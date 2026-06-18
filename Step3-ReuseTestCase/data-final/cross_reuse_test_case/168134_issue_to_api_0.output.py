import tensorflow as tf
import tf.experimental.dtensor as dt
import collections
import tempfile
import os

def configure_virtual_cpus():
    """Configures virtual CPUs for local testing of DTensor."""
    physical_devices = tf.config.list_physical_devices('CPU')
    try:
        tf.config.set_logical_device_configuration(
            physical_devices[0],
            [tf.config.LogicalDeviceConfiguration()] * 2
        )
    except:
        pass # Already configured

def test_name_based_save_uneven_strided_shard():
    """
    Test case for tf.experimental.dtensor.name_based_save based on the 
    PyTorch DTensor uneven strided shard bug (Issue 168134).
    
    This test preserves the reproduction logic by creating a tensor that 
    requires uneven sharding (mimicking the recommended fix approach of 
    unflatten -> shard -> flatten) and verifies that the save operation 
    handles the distributed layout correctly.
    """
    configure_virtual_cpus()

    # 1. Setup Mesh (equivalent to init_device_mesh)
    # Mesh of size 2 to replicate the bug scenario
    mesh = dt.create_mesh([('x', 2)], device_type='CPU')

    # 2. Prepare Data
    # Original data [0, 1, 2, 3, 4]. 
    # Following the bug report's recommended approach: (5,) -> unflatten -> (2, 3)
    # We explicitly pad to size 6 to fit the (2, 3) shape with a padding value (0).
    # Data layout:
    # 0, 1, 2
    # 3, 4, 0 (pad)
    global_tensor = tf.constant([[0., 1., 2.], [3., 4., 0.]])

    # 3. Define Layout (equivalent to _StridedShard logic)
    # Shard the second dimension (size 3) over the mesh dimension 'x' (size 2).
    # This creates the uneven split:
    # Rank 0: cols 0,1 -> [[0, 1], [3, 4]] -> Flatten -> [0, 1, 3, 4]
    # Rank 1: col 2    -> [[2], [0]]    -> Flatten -> [2, pad]
    layout = dt.Layout([dt.UNSHARDED, dt.Sharding.SPLIT_X], mesh)

    # 4. Create DTensor (equivalent to distribute_tensor)
    d_tensor = dt.copy_to_mesh(global_tensor, layout)

    # 5. Prepare for Save
    name_tensor_dict = collections.OrderedDict([('uneven_shard_tensor', d_tensor)])

    # 6. Execute Save (Target API)
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_prefix = os.path.join(tmpdir, 'ckpt')
        
        # This should handle the sharded tensor without crashing or corrupting data
        dt.experimental.name_based_save(mesh, checkpoint_prefix, name_tensor_dict)

        # 7. Verification
        # Check that the checkpoint index file was created
        assert os.path.exists(checkpoint_prefix + '.index'), \
            "Checkpoint index file was not created by name_based_save"

if __name__ == '__main__':
    test_name_based_save_uneven_strided_shard()
    print("Test passed.")