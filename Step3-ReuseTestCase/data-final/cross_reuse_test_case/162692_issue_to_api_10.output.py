import torch
import tensorflow as tf
import tf.experimental.dtensor as dtensor
import numpy as np

def test_dtensor_mean_uneven_sharding():
    """
    Test case adapted from PyTorch Issue #162692.
    Verifies that tf.experimental.dtensor correctly computes the mean
    of a tensor with uneven sharding, leveraging tf.experimental.dtensor.get_default_mesh.
    """
    
    # Initialize the DTensor system
    # PyTorch equivalent: dist.init_process_group(...)
    dtensor.initialize_accelerator_system('cpu')

    # Create a mesh with 2 devices to simulate the uneven sharding scenario
    # PyTorch equivalent: mesh = init_device_mesh('cuda', (2,))
    mesh = dtensor.create_mesh(
        ['x'],
        devices=['CPU:0', 'CPU:1']
    )

    # Set the default mesh context and retrieve it using the similar API
    # PyTorch equivalent: The mesh is passed explicitly to distribute_tensor
    with dtensor.default_mesh(mesh):
        # Leverage the similar API: get_default_mesh
        active_mesh = dtensor.get_default_mesh()
        assert active_mesh is not None, "Default mesh should be accessible"

        # Create the tensor: 3x4
        # PyTorch equivalent: torch.arange(12).reshape(-1, 4).float()
        values = tf.range(12, dtype=tf.float32)
        tensor = tf.reshape(values, (3, 4))

        # Define Layout: Shard on dim 0 (uneven sharding: 2 rows on one device, 1 on the other)
        # PyTorch equivalent: placements=[Shard(0)]
        # We use the mesh retrieved via get_default_mesh to define the layout
        layout = dtensor.Layout([dtensor.Sharding('x'), dtensor.UNSHARDED], active_mesh)

        # Distribute tensor
        # PyTorch equivalent: dt = distribute_tensor(tensor, device_mesh=mesh, placements=[Shard(0)])
        dt = dtensor.DTensor(tensor, layout)

        # Compute mean
        # PyTorch equivalent: mean = dt.mean()
        mean = tf.math.reduce_mean(dt)

        # Expected result: sum(0..11) / 12 = 66 / 12 = 5.5
        expected = 5.5

        # Verify the result
        # The bug in PyTorch resulted in incorrect values or NaN.
        # We assert that the TF implementation handles the uneven sharding correctly.
        result_val = mean.numpy()
        
        print(f"Computed Mean: {result_val}, Expected: {expected}")
        assert np.isclose(result_val, expected), \
            f"Mean calculation incorrect for uneven sharding: {result_val} != {expected}"

if __name__ == '__main__':
    test_dtensor_mean_uneven_sharding()