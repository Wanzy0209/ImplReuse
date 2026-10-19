```python
import tensorflow as tf

# ... (Assuming 'config', 'graph', and 'count' are defined in the surrounding scope)

if config.joint_custom_pre_pass is not None:
    # Conversion: Replaced GraphTransformObserver with direct function call
    # Original: GraphTransformObserver(graph, "joint_custom_pre_pass").apply_graph_pass(...)
    config.joint_custom_pre_pass(graph)
    count += 1

from .post_grad import remove_noop_ops

# Conversion: Replaced GraphTransformObserver with direct function call
# Original: GraphTransformObserver(graph, "remove_noop_ops").apply_graph_pass(remove_noop_ops)
remove_noop_ops(graph)

if config.joint_graph_constant_folding:
    # Conversion: Replaced GraphTransformObserver with direct function call
    # Original: GraphTransformObserver(graph, "constant_fold_uniform_value").apply_gm_pass(...)
    # Note: In TensorFlow, constant folding is typically handled by Grappler or graph_util.
    # We assume constant_fold_uniform_value is a compatible TF graph utility function.
    constant_fold_uniform_value(graph)

if config.joint_custom_pre_pass is not None:
    # Conversion: Replaced GraphTransformObserver with direct function call
    # Original: GraphTransformObserver(graph, "joint_custom_pre_pass").apply_graph_pass(...)
    config.joint_custom_pre_pass(graph)
    count += 1
```