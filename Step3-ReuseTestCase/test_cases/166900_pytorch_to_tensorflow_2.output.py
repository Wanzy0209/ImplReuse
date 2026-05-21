import torch
import tensorflow as tf
from tensorflow.experimental import dtensor

# Define the custom classes from the original bug report
class Foo:
    pass

class Bar:
    def __eq__(self, other):
        return super().__eq__(other)

    def __hash__(self):
        return 0

# Setup necessary for tf.experimental.dtensor.copy_to_mesh
# We use available CPU devices to create a local mesh
devices = tf.config.list_physical_devices('CPU')
if not devices:
    # Fallback for environments where physical devices aren't listed explicitly
    # In a real distributed setup, this would be the actual mesh devices
    devices = [tf.DeviceSpec(device_type="CPU", device_index=0)]

mesh = dtensor.Mesh(
    ['batch'], 
    [dtensor.Mesh.Dimension('batch', len(devices))], 
    devices
)

# Create a replicated layout for the tensor
layout = dtensor.Layout.replicated(mesh, rank=2)

# Adapt the original test logic:
# Instead of @torch.compile, we use @tf.function (the TF equivalent for compilation/tracing).
# We call copy_to_mesh inside, similar to how the original function performed tensor ops.
@tf.function
def fn(x, obj):
    # Preserve the side-effect logic from the original bug
    obj.attr = {3: Bar()}
    
    # Call the similar API: copy_to_mesh
    # This copies the regular tensor 'x' onto the DTensor mesh
    return dtensor.copy_to_mesh(x, layout)

# Create inputs
tensor_input = tf.ones((3, 3))
foo_instance = Foo()

# Execute the function
try:
    result = fn(tensor_input, foo_instance)
    
    # Verify the result is a DTensor and has the correct values
    assert isinstance(result, dtensor.DTensor)
    # Check values (DTensor values can be accessed via .to_tensor() or similar depending on TF version context,
    # but here we just ensure the call didn't crash and returned the correct type)
    print("Test passed: tf.experimental.dtensor.copy_to_mesh handled the context correctly.")
    
except Exception as e:
    print(f"Test failed with error: {e}")
    raise