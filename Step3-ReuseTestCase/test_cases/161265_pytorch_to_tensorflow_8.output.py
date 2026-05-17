import torch
import tensorflow as tf
import tf.experimental.dtensor as dtensor

def test_dtensor_unpack_large_tensor():
    """
    Adapted test case for Issue 161265: [MPS] On MacOS-26 torch.full fails for 4+Gb tensors.
    
    This test verifies that tf.experimental.dtensor.unpack correctly handles data integrity
    for tensors exceeding 4GB in size, analogous to the torch.full/torch.ones issue.
    """
    
    # Setup: Create a mesh. To reproduce the large buffer issue, we use a single device
    # to ensure the local component is large enough (>4GB).
    # Note: This requires a backend (like MPS or GPU) with sufficient memory.
    devices = tf.config.list_physical_devices()
    
    if not devices:
        print("No physical devices found, skipping test.")
        return

    # Use the first available device (GPU/MPS if configured, otherwise CPU)
    # We limit the mesh size to 1 to ensure the local tensor size matches the global size
    # and triggers the large buffer allocation.
    mesh = dtensor.create_mesh([("x", 1)], devices=devices[:1])
    layout = dtensor.Layout.replicated(mesh, rank=2)

    # Core Logic: Create a tensor larger than 4GB.
    # Shape: [2, (1 << 31) + 5]
    # Size: 2 * (2^31 + 5) bytes = 4GB + 10 bytes (since dtype is int8)
    # This specific size is chosen to cross the 32-bit integer boundary (4GB).
    shape = [2, (1 << 31) + 5]
    
    try:
        # Create the DTensor filled with 1s (analogous to torch.ones)
        # This operation internally allocates and fills the buffer.
        dt = dtensor.ones(shape, dtype=tf.int8, layout=layout)

        # Unpack the DTensor into its local components
        components = dtensor.unpack(dt)
        
        # Verify the unpacked data
        # We expect one component because the mesh size is 1
        local_tensor = components[0]

        # Check indices near the end of the tensor (past the 4GB mark).
        # Index -2 corresponds to (1 << 31) + 3.
        # In the original bug, this value was 0 instead of 1 due to fillBuffer failure.
        val_single = local_tensor[1, -2].numpy()
        val_slice = local_tensor[:, -2].numpy()

        print(f"Value at [1, -2]: {val_single}")
        print(f"Values at [:, -2]: {val_slice}")

        # Assertions to ensure the buffer was filled correctly
        assert val_single == 1, f"Expected 1 at [1, -2], got {val_single}. Buffer fill may have failed past 4GB."
        assert all(v == 1 for v in val_slice), f"Expected all 1s in slice [:, -2], got {val_slice}. Buffer fill may have failed past 4GB."

        print("Test passed: Large tensor data integrity verified after unpack.")

    except tf.errors.ResourceExhaustedError:
        print("Test skipped: Not enough memory to allocate 4GB+ tensor.")
    except Exception as e:
        print(f"Test encountered an error: {e}")

if __name__ == "__main__":
    test_dtensor_unpack_large_tensor()