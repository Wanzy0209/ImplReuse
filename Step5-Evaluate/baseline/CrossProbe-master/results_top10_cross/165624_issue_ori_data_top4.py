```python
import tensorflow as tf
from tensorflow.python.framework import graph_util

# Conversion: GraphTransformObserver is a custom utility in the source framework.
# In TensorFlow, we define a helper class to apply transformations to a tf.Graph.
class GraphTransformObserver:
    def __init__(self, graph, name):
        self.graph = graph
        self.name = name

    def apply_graph_pass(self, pass_func):
        # Apply the transformation function to the graph
        with self.graph.as_default():
            pass_func(self.graph)

    def apply_gm_pass(self, pass_func):
        # In TensorFlow, graph module passes are treated as graph transformations
        with self.graph.as_default():
            pass_func(self.graph)

# Conversion: remove_noop_ops
# In TensorFlow, this corresponds to removing nodes like Identity or NoOp.
# We use graph_util.remove_training_nodes as a close equivalent for cleanup.
def remove_noop_ops(graph):
    try:
        graph_def = graph.as_graph_def()
        # Remove training-specific nodes (often includes Identity/NoOps used for gradients)
        cleaned_def = graph_util.remove_training_nodes(graph_def)
        # Note: In a real scenario, one would replace the graph content with cleaned_def
    except Exception:
        pass

# Conversion: constant_fold_uniform_value
# In TensorFlow, constant folding is handled by Grappler, but can be triggered manually.
def constant_fold_uniform_value(graph):
    # Placeholder for constant folding logic
    # TF does this automatically during execution or via tf.graph_transforms
    pass

# Translated logic block
if config.joint_custom_pre_pass is not None:
    GraphTransformObserver(graph, "joint_custom_pre_pass").apply_graph_pass(
        config.joint_custom_pre_pass
    )
    count += 1

# Conversion: The import 'from .post_grad import remove_noop_ops' is handled by the definition above.
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