import torch
"""
Adapted Test Case for TensorFlow DTensor

This test case adapts the PyTorch pipeline parallelism + torch.compile scenario
to TensorFlow's ecosystem. It verifies the behavior of tf.experimental.dtensor.copy_to_mesh
within a compiled execution context (tf.function), mirroring the original issue's
setup of distributed schedules with compiled models.
"""

import tensorflow as tf
import tf.experimental.dtensor as dt

# 1. Setup Distributed Mesh
# Analogous to torch.distributed.device_mesh.init_device_mesh
# We use CPU devices here for general reproducibility, but this applies to GPU/TPU.
try:
    # Try to use available physical devices
    devices = tf.config.list_physical_devices()
    if not devices:
        raise RuntimeError("No devices found")
    # Create a mesh. In the original bug, PP_DEGREE=2. Here we define a mesh dimension.
    mesh = dt.create_mesh([("batch", len(devices))], devices=devices)
except Exception as e:
    print(f"Mesh setup warning: {e}. Falling back to single CPU mesh.")
    mesh = dt.create_mesh([("batch", 1)], devices=[tf.DeviceSpec(device_type="CPU")])


# 2. Define Model
# Analogous to the Transformer class in the original bug report.
class SimpleTransformerLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.linear = tf.keras.layers.Dense(32, use_bias=False)

    def call(self, x):
        return self.linear(x)


# 3. Define Layout
# Defines how the tensor is distributed across the mesh.
# Analogous to the distribution logic in pipeline_module_split.
layout = dt.Layout([dt.UNSHARDED, dt.UNSHARDED], mesh)


# 4. Pipeline Step with Compilation
# Analogous to torch.compile(model) + pp_schedule.step()
# We use @tf.function to compile the graph, similar to torch.compile.
@tf.function
def run_pipeline_stage(input_tensor, model, mesh_layout):
    """
    Simulates a single step in a pipeline parallel schedule.
    It moves the input tensor onto the mesh using copy_to_mesh and executes the model.
    """
    # API Under Test: tf.experimental.dtensor.copy_to_mesh
    # This moves the regular tensor onto the DTensor mesh with the specified layout.
    distributed_input = dt.copy_to_mesh(input_tensor, layout=mesh_layout)
    
    # Execute the compiled model logic
    output = model(distributed_input)
    
    return output


def main():
    # Initialize Model
    model = SimpleTransformerLayer()
    
    # Create dummy input data
    # Analogous to input_ids = torch.randint(...)
    batch_size = 8
    seq_len = 32
    inputs = tf.random.uniform((batch_size, seq_len), minval=0, maxval=128, dtype=tf.float32)

    print(f"Running compiled pipeline step on mesh: {mesh}")
    print(f"Input shape: {inputs.shape}")

    try:
        # Run the step
        # This tests if copy_to_mesh interacts correctly with the compiled graph (tf.function)
        outputs = run_pipeline_stage(inputs, model, layout)
        
        # Assertions to verify correctness
        assert outputs is not None, "Output is None"
        assert isinstance(outputs, dt.DTensor), f"Expected DTensor, got {type(outputs)}"
        assert outputs.shape == (batch_size, 32), f"Shape mismatch: {outputs.shape}"
        
        print("Test Passed: copy_to_mesh works correctly within compiled pipeline step.")
        print(f"Output type: {type(outputs)}, Shape: {outputs.shape}")

    except Exception as e:
        print(f"Test Failed: {e}")
        raise


if __name__ == "__main__":
    main()