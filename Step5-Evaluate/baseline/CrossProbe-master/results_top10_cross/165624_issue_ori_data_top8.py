```python
import tensorflow as tf

# Conversion: Define a helper class to mimic the GraphTransformObserver behavior
# using TensorFlow's tf.Graph manipulation context.
class GraphTransformObserver:
    def __init__(self, graph, name):
        self.graph = graph
        self.name = name

    def apply_graph_pass(self, pass_func):
        # Conversion: Apply the pass function within the graph's default context.
        with self.graph.as_default():
            pass_func(self.graph)

    def apply_gm_pass(self, pass_func):
        # Conversion: 'gm_pass' refers to GraphModule passes in PyTorch.
        # In TensorFlow, this maps to standard graph transformations.
        with self.graph.as_default():
            pass_func(self.graph)

# Conversion: Define the graph optimization functions.
# Note: TensorFlow's Grappler optimizer handles most of these automatically.
# These are manual implementations for the sake of the translation.
def remove_noop_ops(graph):
    # Conversion: Iterate through operations to identify and handle NoOps.
    # tf.graph_util.remove_training_nodes is a similar utility for training graphs.
    for op in graph.get_operations():
        if op.type == 'NoOp':
            # Logic to remove or bypass the op would go here.
            pass

def constant_fold_uniform_value(graph):
    # Conversion: Constant folding is an internal TF optimization.
    # This function represents a manual trigger or specific folding logic.
    pass

# Main logic translation
if config.joint_custom_pre_pass is not None:
    GraphTransformObserver(graph, "joint_custom_pre_pass").apply_graph_pass(
        config.joint_custom_pre_pass
    )
    count += 1

# Conversion: Replacing the relative import with the defined function.
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