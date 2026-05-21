import torch
import tensorflow as tf
import functools

# Adapt the test case for tf.nn.atrous_conv2d.
# The original bug involved a functools.partial'ed callback (context_fn) 
# failing inside a compiled function (torch.compile).
# Since tf.nn.atrous_conv2d does not take a callback, we adapt the logic 
# to test if the API handles functools.partial correctly when used 
# inside a compiled context (tf.function).

def test_atrous_conv2d_with_partial():
    # Define inputs
    # Shape: [batch, height, width, in_channels]
    value = tf.random.normal([1, 5, 5, 3])
    # Shape: [filter_height, filter_width, in_channels, out_channels]
    filters = tf.random.normal([3, 3, 3, 2])

    # Create a partial function for atrous_conv2d
    # This mimics the 'context_fn1 = functools.partial(...)' in the original bug
    conv_op = functools.partial(tf.nn.atrous_conv2d, rate=2, padding='SAME')

    # Wrap in tf.function (equivalent to torch.compile)
    @tf.function
    def compiled_conv(x, f):
        return conv_op(x, f)

    # Run the operation and compute gradients
    with tf.GradientTape() as tape:
        tape.watch(value)
        output = compiled_conv(value, filters)
        loss = tf.reduce_sum(output)

    grads = tape.gradient(loss, value)

    # Verify output shape and gradients
    assert output.shape == (1, 5, 5, 2), f"Expected shape (1, 5, 5, 2), got {output.shape}"
    assert grads is not None, "Gradients should not be None"
    
    print("Test passed: tf.nn.atrous_conv2d works with functools.partial inside tf.function.")

if __name__ == "__main__":
    test_atrous_conv2d_with_partial()