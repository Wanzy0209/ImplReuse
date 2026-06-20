import warnings
import tensorflow as tf

# Mimic the strict warning environment from the bug report
# to ensure no warnings are raised during the conversion process.
warnings.simplefilter('error')

print(tf.__version__)

# Initialize a variable to represent the global step.
# This corresponds to the tensor that needs to be converted to a scalar.
global_step_tensor = tf.Variable(0, dtype=tf.int32, name='global_step')

# Create a session to run the operations.
# This is required because tf.compat.v1.train.global_step expects a session object.
with tf.compat.v1.Session() as sess:
    # Initialize the variable
    sess.run(tf.compat.v1.variables_initializer([global_step_tensor]))

    # Loop structure similar to the bug report's optimization loop.
    for i in range(100):
        # Update the step (mimicking the optimizer step).
        # In graph mode, we must run the assign operation.
        sess.run(global_step_tensor.assign_add(1))

        # Retrieve the scalar value using the similar API.
        # This corresponds to the `float(closure())` call in LBFGS that caused the warning.
        # Pass the session object 'sess' instead of None.
        step_val = tf.compat.v1.train.global_step(sess, global_step_tensor)

        # Print and assert to verify correct behavior.
        print(i, step_val)
        assert step_val == i + 1