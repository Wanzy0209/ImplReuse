import tensorflow as tf
import numpy as np
from tensorflow.experimental import dtensor

# Adapted from PyTorch issue 165861:
# The original bug involves a dimension size exceeding uint16 max (2**16) causing a CUDA error.
# Here we test the robustness of tf.experimental.dtensor.Mesh when handling large dimension sizes.

# Note: Creating a Mesh requires physical devices. 
# This test attempts to define a mesh with a dimension of 2**16.
# We expect this to fail due to resource constraints (lack of 65536 devices) in a standard environment,
# but we verify that the API handles the large integer input gracefully (raises an error) 
# rather than causing an undefined crash or silent failure.

def test_mesh_large_dimension():
    # 1. Test with a small dimension (Control)
    # This should work if at least one device is available.
    devices = tf.config.list_physical_devices()
    if not devices:
        print("No physical devices found. Skipping control test.")
        return

    try:
        # Create a mesh with a small dimension
        mesh_small = dtensor.create_mesh(
            ['batch'],
            devices=devices,
            global_shape=[len(devices)]
        )
        print("Small mesh creation ok")
    except Exception as e:
        print(f"Small mesh creation failed: {e}")

    # 2. Test with a large dimension (2**16)
    # This mirrors the PyTorch condition where a dimension is 2**16.
    large_dim_size = 2**16
    
    print(f"Attempting to create mesh with dimension {large_dim_size}...")
    
    try:
        # We attempt to define a mesh with a dimension of 2**16.
        # We use the available devices list, but request a global shape
        # that implies 65536 devices.
        
        # Note: create_mesh validates that the number of devices matches the global shape.
        # We expect this to raise an error because len(devices) != large_dim_size.
        
        mesh_large = dtensor.create_mesh(
            ['batch'],
            devices=devices,
            global_shape=[large_dim_size]
        )
        print("Large mesh creation ok (Unexpected: implies 65536 devices available)")
        
    except Exception as e:
        # We expect an error here (e.g., InvalidArgumentError regarding device count).
        # We just want to ensure it's a caught exception, not a segfault.
        print(f"Large mesh creation failed (Expected): {e}")

if __name__ == "__main__":
    test_mesh_large_dimension()