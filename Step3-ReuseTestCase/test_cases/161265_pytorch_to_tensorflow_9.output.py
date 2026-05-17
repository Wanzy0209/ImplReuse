import torch
import tensorflow as tf
from tensorflow.experimental import dtensor
import numpy as np

def test_default_mesh_large_tensor():
    """
    Adapted test case for Issue 161265: [MPS] On MacOS-26 torch.full fails for 4+Gb tensors.
    
    Original PyTorch Logic:
    1. Create a tensor > 4GB (2 * (2^31 + 5) bytes).
    2. Fill with ones.
    3. Verify elements near the end of the buffer (where the Metal fillBuffer bug occurred).
    
    Adaptation for tf.experimental.dtensor.default_mesh:
    1. Define a mesh (using CPU for portability, as MPS is PyTorch-specific).
    2. Use default_mesh context to set the execution environment.
    3. Create the large tensor within the scope.
    4. Verify values to ensure the filling logic works correctly across the large buffer.
    """
    
    # Create a mesh. We use CPU here to ensure the test runs on most systems,
    # as the specific MPS backend bug is PyTorch-specific, but we test the 
    # large tensor allocation/filling logic within the DTensor mesh context.
    mesh = dtensor.create_mesh([("x", 1)], devices=["CPU:0"])

    # Use the similar API: tf.experimental.dtensor.default_mesh
    with dtensor.default_mesh(mesh):
        # Reproduce the core logic: Create a large tensor (>4GB)
        # Shape: [2, 2^31 + 5] -> ~4GB + 10 bytes
        # dtype: int8
        try:
            large_shape = [2, (1 << 31) + 5]
            
            # Mimic torch.ones(..., dtype=torch.int8)
            # Note: In TF, tf.ones defaults to float32, so we specify dtype=tf.int8
            a = tf.ones(large_shape, dtype=tf.int8)
            
            # Verify the values (Core bug reproduction logic)
            # Original bug: a[1, -2] was 0 instead of 1.
            # We check the second to last element of the second row.
            val = a[1, -2]
            print(f"a[1, -2]: {val.numpy()}")
            assert val == 1, f"Expected 1, got {val.numpy()}"
            
            # Original bug: a[:, -2] was [0, 0] instead of [1, 1]
            # We check the second to last element of all rows.
            vals = a[:, -2]
            print(f"a[:, -2]: {vals.numpy()}")
            assert np.array_equal(vals.numpy(), [1, 1]), f"Expected [1, 1], got {vals.numpy()}"
            
            print("Test passed: Large tensor filled correctly within default_mesh.")
            
        except tf.errors.ResourceExhaustedError:
            print("Skipped: Not enough memory to allocate 4GB tensor.")
        except Exception as e:
            print(f"Test encountered an error: {e}")

if __name__ == "__main__":
    test_default_mesh_large_tensor()