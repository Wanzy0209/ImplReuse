import tensorflow as tf

# Adapted test case to reproduce the logic of the merge mistake using tf.name_scope

# Mock configuration object similar to PyTorch's config
class Config:
    def __init__(self):
        # Set to a function to trigger the conditional blocks
        self.joint_custom_pre_pass = self.mock_custom_pass
        self.joint_graph_constant_folding = True

    def mock_custom_pass(self, graph):
        """
        Simulates the custom pre-pass logic.
        In the context of tf.name_scope, we create a dummy operation
        to represent the graph transformation.
        """
        with tf.name_scope("custom_pre_pass_internal"):
            # Create a dummy variable to simulate a graph modification
            _ = tf.Variable(1.0, name="dummy_var")
            print("Executed custom_pre_pass")

config = Config()

# Counter to track the number of times the pass is applied
count = 0

# --- Start of adapted logic from joint_graph.py ---

# Block 1: First invocation of joint_custom_pre_pass
if config.joint_custom_pre_pass is not None:
    # Using tf.name_scope to mimic the GraphTransformObserver context
    with tf.name_scope("joint_custom_pre_pass"):
        config.joint_custom_pre_pass(None)
        count += 1

# Intermediate step: remove_noop_ops equivalent
# In TensorFlow, this might be a graph optimization pass, here represented by a scope
with tf.name_scope("remove_noop_ops"):
    pass

# Intermediate step: constant_folding equivalent
if config.joint_graph_constant_folding:
    with tf.name_scope("constant_fold_uniform_value"):
        pass

# Block 2: Second invocation of joint_custom_pre_pass (The Bug)
# This block is the duplicate that should not exist in a fixed version.
if config.joint_custom_pre_pass is not None:
    with tf.name_scope("joint_custom_pre_pass"):
        config.joint_custom_pre_pass(None)
        count += 1

# --- End of adapted logic ---

# Verification
# The bug reproduction logic implies that the code above runs the pass twice.
# Therefore, we assert that count is 2 to confirm the logic matches the buggy behavior.
print(f"Total pass executions: {count}")
assert count == 2, f"Expected 2 executions due to duplicate logic, but got {count}"