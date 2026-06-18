import torch
import tensorflow as tf
import numpy as np

# Set seeds for reproducibility
tf.random.set_seed(0)
np.random.seed(0)

def foo(input_tensor):
    """
    Adapted function using tf.experimental.dtensor.Mesh.
    Instead of interpolation, we test the consistency of creating a Mesh
    and using it to define a DTensor layout, which is the core functionality
    of the Mesh API.
    """
    # Create a Mesh configuration (similar to how interpolate defines a grid)
    # Using a single device mesh for compatibility in standard test environments
    mesh = tf.experimental.dtensor.Mesh(
        dim_names=["x", "y"],
        global_device_ids=np.array([[0]]),
        local_device_ids=[0],
        device_type="CPU"
    )
    
    # Define a layout using the mesh
    layout = tf.experimental.dtensor.Layout([tf.experimental.dtensor.UNSHARDED, tf.experimental.dtensor.UNSHARDED], mesh)
    
    # Create a DTensor from the input using the layout
    # This operation involves the Mesh and should be consistent between eager and compiled modes
    d_tensor = tf.experimental.dtensor.DTensor(input_tensor, layout)
    
    # Perform a reduction (similar to argmin in the original) to get a scalar result
    # to verify computation consistency.
    result = tf.reduce_sum(d_tensor)
    return result

# Generate input data matching the original shape
x = np.random.uniform(0, 10, size=(1, 40, 1, 1))
input_tensor = tf.constant(x, dtype=tf.float64)

# Eager execution
eager_res = foo(input_tensor)

# Compiled execution (tf.function is the TensorFlow equivalent to torch.compile)
cfoo = tf.function(foo)
compile_res = cfoo(input_tensor)

# Assert consistency
np.testing.assert_allclose(eager_res.numpy(), compile_res.numpy())
print("Test passed: Eager and compiled results are consistent.")