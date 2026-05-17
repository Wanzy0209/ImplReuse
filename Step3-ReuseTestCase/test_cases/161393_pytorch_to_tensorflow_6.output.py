import torch
import tensorflow as tf

# Define the function using the similar API: tf.compat.v1.name_scope
def f(x):
    # Use the requested API to create a scope for the operations
    with tf.compat.v1.name_scope("nonzero_slice_scope"):
        # Equivalent of x.nonzero() in PyTorch
        # tf.where returns indices where condition is True, resulting in a dynamic shape
        nz = tf.where(tf.not_equal(x, 0))
        
        # Slicing the tensor with dynamic/unbacked size
        # This corresponds to nz[:-1] in the original bug report
        return nz[:-1]

# Wrap in tf.function to simulate the compilation/graph context
# This is the TensorFlow equivalent of torch.compile for graph execution
compiled_f = tf.function(f)

# Create input tensor
x = tf.random.normal((3, 4))

# Execute the compiled function
out = compiled_f(x)
print(out)