```python
import tensorflow as tf
from tensorflow.python.framework import graph_util

# Conversion: Define GraphTransformObserver to mimic the source structure using TF Graph objects
class GraphTransformObserver:
    def __init__(self, graph, name):
        self.graph = graph
        self.name = name

    def apply_graph_pass(self, pass_func):
        # In TensorFlow, graph passes are typically functions that modify the graph or GraphDef.
        # We assume pass_func accepts the tf.Graph object.
        pass_func(self.graph)

    def apply_gm_pass(self, pass_func):
        # In TensorFlow, there isn't a strict distinction between Graph and GraphModule passes
        # as in PyTorch FX. We treat it as a standard graph transformation.
        pass_func(self.graph)

# Conversion: Define remove_noop_ops using TensorFlow utilities
# The source imports this from .post_grad, so we define it here to ensure validity.
def remove_noop_ops(graph):
    # TensorFlow equivalent: remove_training_nodes
    # This utility removes nodes that are only used for training (like Identity ops).
    # Note: This usually operates on GraphDef, so we convert, transform, and ideally update the graph.
    with graph.as_default():
        graph_def = graph.as_graph_def()
        # We perform the operation. In a real scenario, one might need to import the 
        # modified GraphDef back into a new Graph.
        graph_util.remove_training_nodes(graph_def)

# Conversion: Define constant_fold_uniform_value
def constant_fold_uniform_value(graph):
    # TensorFlow performs constant folding automatically via Grappler or tf.function.
    # Explicitly triggering it requires specific optimization passes or XLA compilation.
    # Here we represent the intent of the pass.
    with graph.as_default():
        # Placeholder for explicit constant folding logic if needed
        pass

# --- Translated Logic ---

if config.joint_custom_pre_pass is not None:
    GraphTransformObserver(graph, "joint_custom_pre_pass").apply_graph_pass(
        config.joint_custom_pre_pass
    )
    count += 1

# from .post_grad import remove_noop_ops
# (Function defined above to replace the import)

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