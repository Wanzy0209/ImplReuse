import tensorflow as tf
import numpy as np

# The similar API (global_variables_initializer) has specific logic for eager vs graph mode.
# To test the initialization functionality effectively (mimicking the "operation" in the bug),
# we disable eager execution. In eager mode, global_variables_initializer is a no-op.
tf.compat.v1.disable_eager_execution()

# Map PyTorch in-place random ops to TensorFlow initializers
# This preserves the logic of testing various random distributions
ops = [
    ("normal_(0,1)", tf.random.normal_initializer(mean=0.0, stddev=1.0)),
    ("uniform_(0,1)", tf.random.uniform_initializer(minval=0.0, maxval=1.0)),
    ("uniform_(0,10)", tf.random.uniform_initializer(minval=0.0, maxval=10.0)),
]

print(f"{'Operation':<20} {'Max Value':<12} {'Status'}")
print("-" * 50)

for name, init_func in ops:
    # Use a new graph for each iteration to ensure clean state
    with tf.compat.v1.Graph().as_default():
        # Create a variable with the specific initializer.
        # Note: While TF variables are contiguous, we test the initialization logic
        # which is the semantic equivalent of the in-place random ops in the bug report.
        var = tf.compat.v1.get_variable(
            f"var_{name}", 
            shape=(50, 50), 
            initializer=init_func
        )

        with tf.compat.v1.Session() as sess:
            # Run the global variables initializer (The API under test)
            sess.run(tf.compat.v1.global_variables_initializer())
            
            # Verify the variable was initialized correctly
            val = sess.run(var)
            max_val = np.max(val)
            
            # Check for silent failure (values remaining 0)
            status = " OK" if max_val != 0.0 else " BUG"
            print(f"{name:<20} {max_val:<12.4f} {status}")