```python
import tensorflow as tf
from tensorflow.python.framework import graph_util

# Conversion: Define a helper class to mimic the source's GraphTransformObserver
# using TensorFlow's graph manipulation APIs.
class GraphTransformObserver:
    def __init__(self, graph, name):
        self.graph = graph
        self.name = name

    def apply_graph_pass(self, pass_func):
        # In TensorFlow, we apply the function directly to the graph or graph_def
        pass_func(self.graph)

    def apply_gm_pass(self, pass_func):
        # Treat graph module passes similarly to graph passes in this context
        pass_func(self.graph)

# Conversion: remove_noop_ops is likely a custom pass in the source.
# We map it to TensorFlow's remove_training_nodes which removes no-ops.
def remove_noop_ops(graph):
    graph_util.remove_training_nodes(graph)

# Conversion: constant_fold_uniform_value maps to TF's constant folding utilities.
def constant_fold_uniform_value(graph):
    # Note: Actual constant folding in TF usually requires a Session and GraphDef.
    # This is a simplified representation of the intent.
    with tf.compat.v1.Session(graph=graph) as sess:
        # Initialize variables if any (dummy init for context)
        sess.run(tf.compat.v1.global_variables_initializer())
        graph_def = graph.as_graph_def()
        # Perform constant folding
        frozen_graph_def = graph_util.convert_variables_to_constants(
            sess, graph_def, []
        )
        # Import back (conceptual, usually returns a new graph)
        tf.import_graph_def(frozen_graph_def, name="")

# --- Translated Logic ---

if config.joint_custom_pre_pass is not None:
    GraphTransformObserver(graph, "joint_custom_pre_pass").apply_graph_pass(
        config.joint_custom_pre_pass
    )
    count += 1

# Conversion: Applying the remove_noop_ops pass
GraphTransformObserver(graph, "remove_noop_ops").apply_graph_pass(remove_noop_ops)

if config.joint_graph_constant_folding:
    GraphTransformObserver(graph, "constant_fold_uniform_value").apply_gm_pass(
        constant_fold_uniform_value
    )

if config.joint_custom_pre_pass is not None:
    GraphTransformObserver(graph, "joint_custom_pre_pass").apply_graph_pass(
        config.joint_custom_pre_pass
    )
    count += 1
```