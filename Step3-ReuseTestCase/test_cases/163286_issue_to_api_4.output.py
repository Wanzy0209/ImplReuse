import tensorflow as tf
import collections
import os
import shutil

# Ensure the experimental namespace is available
try:
    from tensorflow.experimental import dtensor
except ImportError:
    # Fallback for older TF versions or different namespace structures
    from tensorflow.dtensor import dtensor

def test_name_based_restore_dtype_integrity():
    """
    Test case to verify that tf.experimental.dtensor.name_based_restore
    correctly preserves the dtype of tensors, analogous to ensuring
    .view(dtype) is not thrown away during lowering/operations.
    
    This addresses the concern where a tensor viewed or cast as a specific 
    dtype (e.g., uint8) might be incorrectly handled (e.g., reverted to float8).
    """
    # Setup a simple mesh
    mesh = dtensor.create_mesh([("x", 1)], devices=["CPU:0"])

    # Define a checkpoint prefix
    checkpoint_prefix = "./test_ckpt_dir/ckpt"

    try:
        # --- Scenario 1: Float32 Preservation ---
        print("Testing Float32 preservation...")
        with tf.device("/CPU:0"):
            # Create a float32 tensor
            original_float = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
            dt_original_float = dtensor.DTensor(
                original_float, 
                layout=dtensor.Layout([dtensor.ShardingSpec(0, "x")], mesh)
            )

        # Save the float32 tensor
        ckpt_float = tf.train.Checkpoint(tensor=dt_original_float)
        ckpt_float.save(checkpoint_prefix)

        # Prepare to restore into a variable
        with tf.device("/CPU:0"):
            # Initialize a variable with zeros (float32)
            restored_float_var = tf.Variable(initial_value=tf.zeros_like(original_float))
            dt_restored_float = dtensor.DTensor(
                restored_float_var,
                layout=dtensor.Layout([dtensor.ShardingSpec(0, "x")], mesh)
            )

        # Restore
        name_tensor_dict_float = collections.OrderedDict([("tensor", dt_restored_float)])
        dtensor.name_based_restore(mesh, checkpoint_prefix, name_tensor_dict_float)

        # Verify dtype and values
        assert dt_restored_float.dtype == tf.float32, \
            f"Expected dtype float32, but got {dt_restored_float.dtype}"
        assert tf.reduce_all(dt_restored_float._tensor == original_float).numpy(), \
            "Float32 values mismatch after restore."
        print("Float32 preservation successful.")

        # --- Scenario 2: Uint8 Preservation (Mirroring the bug report's context) ---
        print("Testing Uint8 preservation...")
        with tf.device("/CPU:0"):
            # Create a uint8 tensor (simulating the scales/view in the bug report)
            original_uint8 = tf.constant([10, 20, 30], dtype=tf.uint8)
            dt_original_uint8 = dtensor.DTensor(
                original_uint8,
                layout=dtensor.Layout([dtensor.ShardingSpec(0, "x")], mesh)
            )

        # Save the uint8 tensor
        ckpt_uint8 = tf.train.Checkpoint(tensor=dt_original_uint8)
        ckpt_uint8.save(checkpoint_prefix)

        # Prepare to restore
        with tf.device("/CPU:0"):
            restored_uint8_var = tf.Variable(initial_value=tf.zeros_like(original_uint8))
            dt_restored_uint8 = dtensor.DTensor(
                restored_uint8_var,
                layout=dtensor.Layout([dtensor.ShardingSpec(0, "x")], mesh)
            )

        # Restore
        name_tensor_dict_uint8 = collections.OrderedDict([("tensor", dt_restored_uint8)])
        dtensor.name_based_restore(mesh, checkpoint_prefix, name_tensor_dict_uint8)

        # Verify dtype and values
        # The bug in PyTorch was that uint8 was treated as float8. 
        # Here we assert it remains uint8.
        assert dt_restored_uint8.dtype == tf.uint8, \
            f"Expected dtype uint8, but got {dt_restored_uint8.dtype}. " \
            "This mirrors the bug where view(dtype) was ignored."
        assert tf.reduce_all(dt_restored_uint8._tensor == original_uint8).numpy(), \
            "Uint8 values mismatch after restore."
        print("Uint8 preservation successful.")

    finally:
        # Cleanup
        if os.path.exists("./test_ckpt_dir"):
            shutil.rmtree("./test_ckpt_dir")

if __name__ == "__main__":
    test_name_based_restore_dtype_integrity()