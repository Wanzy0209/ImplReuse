```python
import tensorflow as tf

# Conversion: GraphTransformObserver is a custom wrapper to mimic the source structure.
# In TensorFlow, graph transformations are typically applied to GraphDefs or via Grappler.
class GraphTransformObserver:
    def __init__(self, graph, name):
        self.graph = graph
        self.name = name

    def apply_graph_pass(self, pass_func):
        # Apply the transformation function to the graph
        # In TF, this often involves manipulating the GraphDef
        pass_func(self.graph)

    def apply_gm_pass(self, pass_func):
        # Conversion: 'gm_pass' (GraphModule pass) in PyTorch maps to graph-level transformations in TF.
        # We treat it similarly to apply_graph_pass here.
        pass_func(self.graph)

# Conversion: remove_noop_ops maps to removing training nodes (e.g., Assert, CheckNumerics)
# which are effectively NoOps in inference mode.
def remove_noop_ops(graph):
    # TensorFlow equivalent: tf.compat.v1.graph_util.remove_training_nodes
    # Note: This usually operates on a GraphDef, so we might need to convert depending on the graph object type.
    if isinstance(graph, tf.Graph):
        # Placeholder for actual graph manipulation logic
        pass

# Conversion: constant_fold_uniform_value maps to constant folding operations.
# TensorFlow performs this automatically via Grappler, but can be done manually via graph_util.
def constant_fold_uniform_value(graph):
    # TensorFlow equivalent: tf.compat.v1.graph_util.convert_variables_to_constants
    # or relying on the optimizer to fold constants.
    pass

# Main logic block
if config.joint_custom_pre_pass is not None:
    GraphTransformObserver(graph, "joint_custom_pre_pass").apply_graph_pass(
        config.joint_custom_pre_pass
    )
    count += 1

# from .post_grad import remove_noop_ops
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