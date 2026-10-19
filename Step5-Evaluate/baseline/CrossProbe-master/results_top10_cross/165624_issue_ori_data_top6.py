```python
import tensorflow as tf
from tensorflow.python.framework import graph_util

# Conversion: Define the GraphTransformObserver class to mimic the source structure
# using TensorFlow graph utilities.
class GraphTransformObserver:
    def __init__(self, graph, name):
        self.graph = graph
        self.name = name

    def apply_graph_pass(self, pass_func):
        # In TensorFlow, graph passes usually operate on GraphDef.
        # We apply the function and update the graph if necessary.
        # Note: Direct graph mutation in TF is limited; usually we create a new GraphDef.
        if callable(pass_func):
            # Assuming pass_func handles the transformation logic
            pass_func(self.graph)

    def apply_gm_pass(self, pass_func):
        # Similar to apply_graph_pass, but typically for Graph Module (GM) level passes.
        # In TF, this maps to Grappler optimizations or GraphDef transformations.
        if callable(pass_func):
            pass_func(self.graph)

# Conversion: Define remove_noop_ops using TensorFlow's graph_util
def remove_noop_ops(graph):
    # TensorFlow equivalent: remove_training_nodes removes ops not needed for inference
    # (e.g., Identity, CheckpointV2).
    # Note: This returns a GraphDef, so the graph object needs to be updated in a real scenario.
    return graph_util.remove_training_nodes(graph.as_graph_def())

# Conversion: Define constant_fold_uniform_value using TensorFlow's graph_util
def constant_fold_uniform_value(graph):
    # TensorFlow equivalent: convert_variables_to_constants folds variables into constants.
    # This typically requires a session and output node names.
    # Here we define the placeholder for the logic.
    # graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), output_node_names)
    pass

# --- Translated Source Code ---

if config.joint_custom_pre_pass is not None:
    GraphTransformObserver(graph, "joint_custom_pre_pass").apply_graph_pass(
        config.joint_custom_pre_pass
    )
    count += 1

# from .post_grad import remove_noop_ops
# Conversion: Using the defined TF equivalent function
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