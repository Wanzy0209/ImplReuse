import torch
import unittest
import tensorflow as tf
import tf.experimental.dtensor as dtensor

class TestDTensorCopyToMesh(unittest.TestCase):
    def test_copy_to_mesh_correctness(self):
        # Check for GPU availability, similar to @requires_gpu in PyTorch
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            self.skipTest("Test requires GPU")

        # Setup Mesh and Layout
        # Using a single device mesh to mimic the single GPU context of the original bug
        mesh = dtensor.create_mesh(['x'], gpus)
        layout = dtensor.Layout([dtensor.UNSHARDED], mesh)

        # Inputs
        x = tf.random.normal((4, 4))
        y = tf.random.normal((4, 4))

        # Eager execution
        # 1. Compute (Simulating the user-defined kernel logic)
        local_out = x + y
        # 2. Transfer (Simulating the device transfer, here Local -> Mesh)
        d_out = dtensor.copy_to_mesh(local_out, layout)
        # 3. Post-transfer operation
        eager_result = d_out + 1

        # Compiled execution (Simulating torch.compile)
        @tf.function
        def f(x, y):
            local_out = x + y
            d_out = dtensor.copy_to_mesh(local_out, layout)
            return d_out + 1

        compiled_result = f(x, y)

        # Verify correctness: Ensure compiled graph matches eager execution
        self.assertAllClose(eager_result, compiled_result)

if __name__ == '__main__':
    unittest.main()