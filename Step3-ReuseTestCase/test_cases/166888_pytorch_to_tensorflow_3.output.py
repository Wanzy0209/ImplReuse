import tensorflow as tf

# Enable eager execution as requested by the similar API.
# This is the TensorFlow equivalent of setting the execution mode,
# though it enables immediate execution rather than graph compilation.
tf.compat.v1.enable_eager_execution()

def f(x, max_val):
    # In the original PyTorch bug, .item() is used to extract a scalar value
    # from a tensor argument to be used in a control flow or arithmetic operation.
    # In TensorFlow eager mode, .numpy() provides the equivalent functionality.
    y = tf.clip_by_value(x, 0.0, max_val.numpy())
    return y

# Create inputs similar to the original test case
x = tf.random.normal((10, 20, 30))
max_val = tf.constant(5.0)

# Execute the function
result = f(x, max_val)

# Verify the behavior
# Check shape
assert result.shape == (10, 20, 30), "Output shape mismatch"

# Check that clamping logic worked (values should be between 0 and 5)
assert tf.reduce_all(result >= 0.0), "Minimum value constraint failed"
assert tf.reduce_all(result <= 5.0), "Maximum value constraint failed"

print("Test passed: Function executed successfully with eager execution.")