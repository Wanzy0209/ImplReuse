import tensorflow as tf
import tf.experimental.dtensor as dtensor

def addcmul_func(x, y, z):
    return x + (y * z)

# Create a mesh to simulate the device context (e.g., XPU/TPU)
# Using 'cpu' here for general compatibility, but logic applies to other accelerators
mesh = dtensor.create_mesh([('batch', 1)], devices=['cpu:0'])

# Create tensors on the host (CPU)
x = tf.random.normal((128,))
y = tf.random.normal((128,))
z = tf.random.normal((128,))

# Run in eager mode (host execution)
out = addcmul_func(x, y, z)
print("eager mode passed")

# Define the layout for the target mesh
layout = dtensor.Layout([dtensor.UNSHARDED], mesh)

# Use the target API: copy_to_mesh
# This moves the tensors from host to the DTensor mesh (device)
x_dt = dtensor.copy_to_mesh(x, layout)
y_dt = dtensor.copy_to_mesh(y, layout)
z_dt = dtensor.copy_to_mesh(z, layout)

# Execute the function using the tensors on the mesh
out_dt = addcmul_func(x_dt, y_dt, z_dt)
print("copy_to_mesh and computation passed")