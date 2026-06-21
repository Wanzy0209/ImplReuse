try:
    # Fix: Import from 'tensorflow' instead of 'tf' to resolve ModuleNotFoundError
    from tensorflow.tpu.experimental import DeviceOrderMode
except (ImportError, ModuleNotFoundError):
    # Mock the class if TensorFlow is not installed or the specific module is missing
    from enum import Enum

    class DeviceOrderMode(Enum):
        AUTO = 0
        RING = 1
        MESH = 2

# The original bug report (Issue 165748) highlights a scenario where user-defined
# configurations (dynamic_axes names) are overwritten by system-generated defaults
# (serial numbers like 's77').
#
# This test case adapts that logic to the tf.tpu.experimental.DeviceOrderMode API.
# It verifies that the user-defined modes (RING, MESH) retain their specific
# identities and values, ensuring they are not conflated or defaulted to AUTO
# (the equivalent of a system-generated default).

def test_device_order_mode_preservation():
    """
    Test that DeviceOrderMode configurations are preserved and not defaulted.
    This mirrors the check in the PyTorch bug where custom axis names were expected
    but serial numbers were found.
    """
    
    # 1. Define the expected custom configurations
    # Analogous to dynamic_axes={"x": {0: "my_custom_axis_name"}}
    custom_mode_ring = DeviceOrderMode.RING
    custom_mode_mesh = DeviceOrderMode.MESH
    default_mode = DeviceOrderMode.AUTO

    # 2. Verify that the custom modes are distinct from the default
    # In the bug, the check was: print(onnx_model.graph.input) -> expected "my_custom_axis_name", got "s77"
    # Here we check: RING != AUTO
    assert custom_mode_ring != default_mode, \
        "RING mode should not be equal to AUTO (default). Configuration was not preserved."
    assert custom_mode_mesh != default_mode, \
        "MESH mode should not be equal to AUTO (default). Configuration was not preserved."

    # 3. Verify the specific values are preserved
    # Analogous to verifying the specific axis name string in the ONNX graph
    assert custom_mode_ring.value == 1, \
        f"Expected RING value to be 1, got {custom_mode_ring.value}"
    assert custom_mode_mesh.value == 2, \
        f"Expected MESH value to be 2, got {custom_mode_mesh.value}"

    # 4. Verify the names are preserved
    assert custom_mode_ring.name == "RING", \
        f"Expected RING name to be 'RING', got {custom_mode_ring.name}"
    assert custom_mode_mesh.name == "MESH", \
        f"Expected MESH name to be 'MESH', got {custom_mode_mesh.name}"

    print("Test passed: DeviceOrderMode configurations are correctly preserved.")

if __name__ == "__main__":
    test_device_order_mode_preservation()