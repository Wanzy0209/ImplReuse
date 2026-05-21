import torch
import tensorflow as tf
import tf.experimental.dtensor as dtensor

# Adaptation of the bug reproduction logic for tf.experimental.dtensor.copy_to_mesh
# Original bug: torch.compile triggers recompile.
# Adapted test: Verify that copy_to_mesh works correctly within a compiled context (tf.function)
# without causing unnecessary retracing (recompilation).

def test_copy_to_mesh_recompile():
    # Setup: Create a mesh (equivalent to .to('cuda'))
    # We use CPU devices to ensure the test runs in most environments
    devices = tf.config.list_physical_devices('CPU')
    if not devices:
        # Fallback if no physical devices are found
        devices = [tf.DeviceSpec(job="localhost", replica=0, task=0, device_type="CPU", device_index=0)]
    
    mesh = dtensor.create_mesh([("batch", 1)], devices=devices)
    
    # Define a layout
    layout = dtensor.Layout([dtensor.UNSHARDED], mesh)

    # Define the compiled function (equivalent to pipe.transformer.compile_repeated_blocks())
    # We use tf.function to simulate the compilation context.
    @tf.function
    def run_computation(tensor):
        # Use the similar API: copy_to_mesh
        # This is the critical step where data is moved to the mesh.
        dtensor_input = dtensor.copy_to_mesh(tensor, layout)
        
        # Simulate inference (simple operation)
        result = dtensor_input * 2.0
        return result

    # Prepare input data
    input_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])

    # Run 1: Initial execution (tracing/compilation happens here)
    print("Running first inference...")
    output_1 = run_computation(input_tensor)

    # Run 2: Subsequent execution (should use cached graph, no recompile)
    print("Running second inference...")
    output_2 = run_computation(input_tensor)

    # Assertions
    # Verify the output is correct
    expected = input_tensor * 2.0
    # Note: output_1 and output_2 are DTensors, we compare their values
    assert tf.reduce_all(output_1 == output_2).numpy(), "Outputs should be consistent"
    assert tf.reduce_all(output_1 == expected).numpy(), "Output calculation is incorrect"
    
    print("Test completed successfully.")

if __name__ == "__main__":
    test_copy_to_mesh_recompile()