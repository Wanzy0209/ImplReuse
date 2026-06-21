import tensorflow as tf
import collections
import tempfile
import os

# Attempt to import dtensor, handle older TensorFlow versions where it is not available
try:
    from tensorflow.experimental import dtensor
except ImportError:
    dtensor = None

if dtensor is None:
    # If dtensor is not available, define a dummy test class that skips the test.
    # This prevents the ImportError and signals the environment limitation.
    class DTensorRestoreTest(tf.test.TestCase):
        @tf.test.skip("DTensor is not available in this TensorFlow version (requires TF >= 2.9).")
        def test_uneven_strided_shard_restore(self):
            pass
else:
    class DTensorRestoreTest(tf.test.TestCase):

      def test_uneven_strided_shard_restore(self):
        """
        Test case adapted from PyTorch Issue 168134.
        
        The original issue describes a bug in uneven strided sharding where a 1D tensor
        [0, 1, 2, 3, 4] was conceptually unflattened to (2, 3), sharded on dim 1,
        and then flattened. This resulted in incorrect local tensor indices.
        
        This test verifies that tf.experimental.dtensor.name_based_restore correctly
        handles the restoration of tensors onto a mesh with a layout that results
        in an uneven split, ensuring the local tensor data matches the expected
        sharding logic.
        """
        # Setup mesh with 2 devices to mimic the bug report's mesh (2,)
        # Using CPU devices for simplicity in testing
        mesh = dtensor.Mesh(['CPU', 'CPU'], [0, 1])

        # Create a global tensor that mimics the unflattened state in the bug report:
        # (5,) -> unflatten -> (2, 3)
        # Data:
        # 0, 1 | 2
        # 3, 4 | 5 (using 5 instead of pad for the checkpoint data)
        global_data = tf.constant([[0, 1, 2], [3, 4, 5]], dtype=tf.float32)

        # Save to checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
          prefix = os.path.join(tmpdir, 'ckpt')
          checkpoint = tf.train.Checkpoint(tensor=global_data)
          save_path = checkpoint.save(prefix)

          # Define the layout that mimics the "StridedShard" behavior:
          # Shard on the second dimension (dim 1) which has size 3.
          # With 2 devices, this creates an uneven split (2 vs 1).
          # This corresponds to the "S(1)" step in the bug description.
          layout = dtensor.Layout([dtensor.UNSHARDED, dtensor.SHARDED], mesh)

          # Initialize DTensor variables with the target layout
          # Rank 0 expects shape (2, 2), Rank 1 expects shape (2, 1)
          # We initialize them to zeros to verify restoration overwrites them correctly
          var_0 = dtensor.DVariable(
              dtensor.zeros(mesh, layout=layout, dtype=tf.float32, shape=(2, 3))
          )

          # Prepare the dictionary for name_based_restore
          # The API expects an OrderedDict of names to DTensors
          name_tensor_dict = collections.OrderedDict([('tensor', var_0)])

          # Perform the restore
          restored_dict = dtensor.name_based_restore(mesh, save_path, name_tensor_dict)

          # Verify the restored tensor
          restored_tensor = restored_dict['tensor']

          # Check local tensors based on the expected uneven split
          # Rank 0 should have [[0, 1], [3, 4]]
          # Rank 1 should have [[2], [5]]
          if mesh.local_device_ids()[0] == 0:
            expected = tf.constant([[0., 1.], [3., 4.]])
            self.assertAllEqual(restored_tensor._local_tensor, expected)
          elif mesh.local_device_ids()[0] == 1:
            expected = tf.constant([[2.], [5.]])
            self.assertAllEqual(restored_tensor._local_tensor, expected)

if __name__ == '__main__':
  tf.test.main()