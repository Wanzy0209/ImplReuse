import tensorflow as tf
import types

# Fix for missing tf.keras.ops in older TensorFlow versions
if not hasattr(tf.keras, 'ops'):
    # Create a mock module for ops
    mock_ops = types.ModuleType('ops')
    # Map the add operation to the standard TensorFlow add
    mock_ops.add = tf.add
    # Assign the mock module to tf.keras.ops
    tf.keras.ops = mock_ops

# Create a tensor with a specific non-contiguous stride to mimic the bug scenario.
# We use slicing to create a view with a stride of 2.
base_tensor = tf.range(10, dtype=tf.float32)
input_tensor = base_tensor[::2]

# Create a second tensor compatible for addition.
other_tensor = tf.ones_like(input_tensor)

# Call the similar API (tf.keras.ops.add)
output_tensor = tf.keras.ops.add(input_tensor, other_tensor)

# Check if the API respects the input stride, mirroring the original bug assertion.
# Note: In TensorFlow, strides are accessed via the experimental numpy interface.
input_stride = tf.experimental.numpy.strides(input_tensor)
output_stride = tf.experimental.numpy.strides(output_tensor)

assert input_stride == output_stride, \
    f"tf.keras.ops.add does not respect input stride. Input: {input_stride}, Output: {output_stride}"