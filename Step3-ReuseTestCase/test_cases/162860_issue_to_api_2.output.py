import torch
import tensorflow as tf

# The original issue involves torch.compile which creates a graph context.
# The similar API (global_variables_initializer) has specific logic for graph mode.
# We disable eager execution to test the API in the context most similar to the bug report.
tf.compat.v1.disable_eager_execution()

def inner(x):
    # Mirrors the inner function in the PyTorch issue
    return x + 1

def test_global_variables_initializer():
    # Setup: Create a variable (analogous to the tensor input in the PyTorch issue)
    # Using tf.ones_initializer to match torch.ones(3)
    var = tf.compat.v1.get_variable(
        "x", shape=[3], initializer=tf.ones_initializer()
    )

    # Logic: Replicate the nested calls from the original issue
    # x = inner(x) -> return inner(x)
    # In TF graph mode, this builds the computation graph.
    y = inner(var)
    result = inner(y)

    # API Under Test: global_variables_initializer
    # This is the candidate for reuse to ensure variables are ready for execution.
    init_op = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        # Run the initializer to realize the variables
        sess.run(init_op)

        # Execute the graph to get the realized value
        output = sess.run(result)

        # Verification: The PyTorch issue requested debug info (type, realized value).
        # We assert these properties here to ensure the variable state is correct.
        assert output.shape == (3,), "Shape should be (3,)"
        assert output.dtype == tf.float32, "Type should be float32"
        # Logic check: 1 (init) + 1 (inner) + 1 (inner) = 3
        assert all(output == 3.0), f"Expected all 3.0, got {output}"

if __name__ == "__main__":
    test_global_variables_initializer()