import torch
import tensorflow as tf
import tensorflow.experimental.dtensor as dtensor

# Define the user-defined class similar to the PyTorch example
class Config:
    def __repr__(self):
        return "Config()"

# Setup for DTensor (required for copy_to_mesh)
# We use available devices to create a mesh
devices = tf.config.list_physical_devices()
if not devices:
    # Fallback for environments without configured devices
    # Note: DTensor typically requires configured devices, 
    # but this ensures the code structure is valid.
    devices = [tf.DeviceSpec(device_type="CPU", device_index=0)]

mesh = dtensor.create_mesh([("batch", len(devices))], devices=devices)
layout = dtensor.Layout([dtensor.UNSHARDED], mesh)

# Define the forward function
# We use @tf.function as the TensorFlow equivalent of torch.compile
# to enable tracing/compilation.
@tf.function
def forward(x, config):
    # Calling repr() on non-constant user object
    # This logic mirrors the original bug report.
    # We verify if the tracer handles this correctly.
    repr_str = repr(config)
    scalar_factor = len(repr_str)
    
    # Call the similar API: copy_to_mesh
    # This moves the tensor to the DTensor mesh
    dtensor_x = dtensor.copy_to_mesh(x, layout)
    
    # Combine the result with the repr logic
    return dtensor_x * scalar_factor

# Instantiate objects
config = Config()
x = tf.random.normal((2, 2))

# Execute the traced function
try:
    result = forward(x, config)
    
    # Verify the output shape and type
    assert isinstance(result, dtensor.DTensor)
    assert result.shape == (2, 2)
    
    print("Test passed: repr() traced successfully with copy_to_mesh.")
except Exception as e:
    print(f"Test failed: {e}")