import torch
import tensorflow as tf
import numpy as np
from tensorflow.experimental.dtensor import Mesh

# Adaptation of the negative padding test to tf.experimental.dtensor.Mesh
# The original bug involves passing negative values to padding arguments.
# Here, we test passing negative values to global_device_ids (which define the mesh dimensions).

def test_mesh_negative_ids():
    # PyTorch case: torch.ops.aten.constant_pad_nd.default(torch.ones([5, 3]), [-1, -2])
    # This attempts to apply negative padding.
    # We attempt to create a Mesh with negative device IDs, which is semantically similar 
    # to passing negative dimension arguments.

    # Define a mesh configuration with negative IDs, mimicking the [-1, -2] padding
    # Shape (2, 2) implies 2 dimensions, but we use negative IDs.
    global_device_ids = np.array([[-1, -2], [0, 1]])
    dim_names = ['x', 'y']
    
    # In PyTorch, negative padding is conditionally allowed.
    # In TensorFlow dtensor, negative device IDs should likely be invalid.
    # We check if the API handles this gracefully or crashes.
    try:
        mesh = Mesh(
            dim_names=dim_names,
            global_device_ids=global_device_ids,
            local_device_ids=[0, 1]
        )
        print("Mesh created with negative IDs (Unexpected behavior)")
        print(f"Mesh shape: {mesh.shape()}")
    except (ValueError, tf.errors.InvalidArgumentError, Exception) as e:
        print(f"Mesh creation failed as expected with negative IDs: {e}")

if __name__ == "__main__":
    test_mesh_negative_ids()