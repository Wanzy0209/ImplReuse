import torch
import tensorflow as tf

# Check execution context, analogous to checking torch.backends.mps.is_available()
# The similar API implementation specifically branches on this condition.
is_eager = tf.executing_eagerly()

# To properly test the initializer (which returns a no-op in eager mode),
# we disable eager execution, similar to requiring a specific device in PyTorch.
if is_eager:
    tf.compat.v1.disable_eager_execution()

# Wrapper over custom variables, mimicking the MPSSoftshrink class structure.
class CustomTFModel:
    def __init__(self):
        # Define variables representing the custom operation's state
        self.var = tf.compat.v1.get_variable("custom_var", shape=[5, 5], initializer=tf.ones_initializer())

    def setup(self):
        # Use the similar API: global_variables_initializer
        # This mirrors the setup of the MPS device and custom kernel in the original issue.
        return tf.compat.v1.global_variables_initializer()

# Reproduce the logic: Check -> Define -> Initialize -> Run
with tf.compat.v1.Session() as sess:
    model = CustomTFModel()
    init_op = model.setup()
    
    # Run the initializer
    sess.run(init_op)
    
    # Verify
    val = sess.run(model.var)
    assert val.shape == (5, 5)