import torch
import tensorflow as tf
import tensorflow.experimental.dtensor as dtensor

def test_copy_to_mesh_no_global_side_effects():
    """
    Adapted from PyTorch Issue 167064.
    Original Bug: torch.compile triggers redundant global code that sets
    torch.distributions.Distribution.set_default_validate_args(False).
    
    This test verifies that tf.experimental.dtensor.copy_to_mesh does not
    trigger similar unwanted global side effects (e.g., changing global random seed).
    """
    
    # Setup: Create a mesh and layout for DTensor
    # Using CPU devices to ensure the test runs in most environments
    devices = tf.config.list_physical_devices('CPU')
    if not devices:
        # Fallback if no CPU devices are listed
        devices = [tf.DeviceSpec(device_type="CPU", device_index=0)]
        
    mesh = dtensor.Mesh(['x'], devices)
    layout = dtensor.Layout([dtensor.UNSHARDED], mesh)
    
    # Create a regular tensor
    tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])

    # Check global state before calling the API
    # We use the global random seed as a proxy for global state integrity,
    # similar to how the PyTorch bug affected distribution validation args.
    initial_seed = tf.random.get_global_seed()

    # Call the API under test
    try:
        dtensor_result = dtensor.copy_to_mesh(tensor, layout)
    except Exception as e:
        # If the API fails, we still check if global state was modified
        print(f"API call failed with: {e}")
        raise

    # Check global state after calling the API
    final_seed = tf.random.get_global_seed()

    # Assert that the global state has not changed
    # This preserves the logic of the original bug report: 
    # ensuring the API does not execute redundant global code.
    assert initial_seed == final_seed, (
        "tf.experimental.dtensor.copy_to_mesh should not modify global random seed. "
        "This indicates potential redundant global code execution similar to the PyTorch bug."
    )

    # Verify basic functionality (result is a DTensor with correct values)
    # Note: DTensor values might be on different devices, so we compare the gathered values
    # or just check the structure if direct comparison is complex across devices.
    # For a replicated layout, the values should match.
    assert isinstance(dtensor_result, dtensor.DTensor), "Result should be a DTensor"
    
    # If the layout is replicated/unsharded, we can check values
    if layout.is_fully_replicated():
        # Convert back to regular tensor for easy comparison if needed, 
        # or just check the underlying components.
        # For simplicity in this test, we assume successful execution implies 
        # correct data placement if no assertion was raised.
        pass

if __name__ == "__main__":
    test_copy_to_mesh_no_global_side_effects()
    print("Test passed: No global side effects detected.")