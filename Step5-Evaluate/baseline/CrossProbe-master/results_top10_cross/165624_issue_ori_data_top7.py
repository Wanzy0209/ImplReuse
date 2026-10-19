```python
import tensorflow as tf

# Conversion Note: In TensorFlow, graph optimizations like removing no-ops and constant folding
# are typically handled automatically by the Grappler optimizer during session execution
# or tf.function compilation. The functions below are structural placeholders to preserve
# the logic of the original source code.

def remove_noop_ops(graph):
    """
    Placeholder for removing no-op operations.
    TensorFlow Equivalent: Handled by the 'pruning' optimizer in Grappler.
    """
    # In a real implementation, one might use tf.graph_editor to manipulate the graph,
    # but typically this is left to the compiler.
    return graph

def constant_fold_uniform_value(graph):
    """
    Placeholder for constant folding.
    TensorFlow Equivalent: Handled by the 'constant folding' optimizer in Grappler.
    """
    return graph

class GraphTransformObserver:
    """
    Helper class to mimic the source's graph transformation interface.
    Interacts with a tf.Graph instance.
    """
    def __init__(self, graph, name):
        self.graph = graph
        self.name = name

    def apply_graph_pass(self, pass_func):
        # Apply the transformation function to the graph
        pass_func(self.graph)

    def apply_gm_pass(self, pass_func):
        # In PyTorch this applies to a GraphModule. In TF, we apply to the Graph.
        pass_func(self.graph)

# Main logic block
if config.joint_custom_pre_pass is not None:
    GraphTransformObserver(graph, "joint_custom_pre_pass").apply_graph_pass(
        config.joint_custom_pre_pass
    )
    count += 1

# Note: remove_noop_ops is applied here
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