import tensorflow as tf

# Ensure we are in TF2 eager mode (default)
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()

# Create a global variable to ensure global_variables() is not empty
# This mimics the tensor setup in the original PyTorch issue
var = tf.Variable(0.0, name="test_var")

# Test Case 1: Eager Execution
# Based on the similar API snippet, this should return a no_op
# because context.executing_eagerly() is True.
eager_init_op = tf.compat.v1.global_variables_initializer()

# Test Case 2: Graph Execution
# We use tf.compat.v1.Graph context to simulate graph mode execution.
# Inside this context, context.executing_eagerly() returns False.
# This avoids the tf.function error where returning an Operation is not allowed.
with tf.compat.v1.Graph().as_default():
    # We need a variable inside this graph to ensure the initializer has dependencies
    var_graph = tf.Variable(0.0, name="test_var_graph")
    graph_init_op = tf.compat.v1.global_variables_initializer()

# Verification
# The original bug highlights a divergence between eager and compile modes.
# The similar API handles this by explicitly branching based on execution mode.
# We verify that the branching logic works as described in the snippet:
# - Eager mode returns a no_op (no inputs).
# - Graph mode returns an initializer op (has inputs/dependencies).

# Check inputs to distinguish NoOp from VariablesInitializer
# A NoOp typically has 0 inputs, while an initializer op depends on variables.
assert len(eager_init_op.inputs) == 0, "Eager mode should return a no_op (no inputs)"
assert len(graph_init_op.inputs) > 0, "Graph mode should return an initializer op (has inputs)"

print(" Test passed: API correctly branches between eager and graph modes.")