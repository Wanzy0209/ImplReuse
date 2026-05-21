import torch
import tensorflow as tf
import tf.experimental.dtensor as dtensor
import unittest
import os

# Disable GPU for this test to ensure it runs in most CI environments
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class TestDTensorCopyToMesh(unittest.TestCase):
    """
    Adapted test case for tf.experimental.dtensor.copy_to_mesh based on 
    PyTorch issue #163594 (test_dtensor_compile_redistribute).
    
    The original issue involved a timeout/hang when using torch.compile
    with DTensor redistribution. This test mimics that scenario by wrapping
    the DTensor operation in tf.function (TensorFlow's graph compilation)
    and executing a redistribution (copy_to_mesh with a specific layout).
    """

    @classmethod
    def setUpClass(cls):
        # Configure virtual devices to simulate a multi-device mesh
        physical_devices = tf.config.list_physical_devices('CPU')
        if physical_devices:
            try:
                tf.config.set_logical_device_configuration(
                    physical_devices[0],
                    [tf.config.LogicalDeviceConfiguration()] * 2
                )
            except RuntimeError:
                # Logical device configuration might already be set or not supported
                pass

    def test_copy_to_mesh_compile_redistribute(self):
        # Create a mesh with 2 logical CPUs
        mesh = dtensor.create_mesh(['x'], ['CPU:0', 'CPU:1'])
        
        # Create a regular tensor
        # Shape: (4, 2)
        tensor = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        
        # Define a sharded layout
        # We shard the first dimension (rows) across the mesh dimension 'x'
        # This mimics the redistribution logic in the original PyTorch test.
        layout = dtensor.Layout([dtensor.SHARDED, dtensor.UNSHARDED], mesh)
        
        # Wrap the operation in tf.function to mimic torch.compile
        # This triggers graph tracing and optimization, which is where
        # the original PyTorch bug (timeout) occurred.
        @tf.function
        def compiled_copy_to_mesh(t):
            return dtensor.copy_to_mesh(t, layout)
        
        # Execute the compiled function
        # If a similar bug exists in TF, this step might hang or timeout.
        result = compiled_copy_to_mesh(tensor)
        
        # Verify the result is a DTensor
        self.assertIsInstance(result, dtensor.DTensor)
        
        # Verify the layout matches the target layout
        self.assertEqual(result.layout, layout)
        
        # Verify the shape is preserved
        self.assertEqual(result.shape, tensor.shape)
        
        # Verify values (conceptually, checking that data is not corrupted)
        # Note: Direct comparison might fail if not handled correctly in distributed context,
        # but checking the structure is the primary goal here.
        self.assertAllClose(tf.identity(result._tensor), tensor)

if __name__ == '__main__':
    unittest.main()