import tensorflow as tf

# The core bug reproduction logic involves passing an extremely large integer
# (9223372036854775803) to a parameter expecting a size or dimension.
# In the original PyTorch case, this was the 'padding' parameter of LazyConv1d.
# Here, we adapt this logic to the TensorFlow API by passing the huge integer
# to the 'mesh_dims' parameter, which defines the size of the mesh dimensions.

huge_size = 9223372036854775803

try:
    # Attempt to create a distributed mesh with a dimension size equal to the huge integer.
    # This tests if the API handles boundary values gracefully or crashes (Segfault).
    mesh = tf.experimental.dtensor.create_distributed_mesh(
        mesh_dims=[('x', huge_size)],
        device_type='CPU'
    )
    print(f"Mesh created: {mesh}")
except Exception as e:
    # A robust API should raise a ValueError or similar error instead of crashing.
    print(f"Caught exception: {type(e).__name__}: {e}")