```python
import tensorflow as tf

# Conversion: GraphTransformObserver is a PyTorch/IR concept. 
# In TensorFlow, graph transformations are applied directly to the GraphDef or via tf.function.
# We assume 'graph' is a tf.compat.v1.GraphDef or similar mutable structure here.

if config.joint_custom_pre_pass is not None:
    # Apply custom pre-pass directly to the graph
    graph = config.joint_custom_pre_pass(graph)
    count += 1

# from .post_grad import remove_noop_ops
# Note: remove_noop_ops is assumed to be a function available in the scope or imported from a TF utility module.
# Standard TF equivalent might be tf.compat.v1.graph_util.remove_training_nodes.
graph = remove_noop_ops(graph)

if config.joint_graph_constant_folding:
    # Apply constant folding pass
    # Standard TF equivalent: tf.compat.v1.graph_util.fold_constants
    graph = constant_fold_uniform_value(graph)

if config.joint_custom_pre_pass is not None:
    # Re-applying custom pre-pass (preserving source logic)
    graph = config.joint_custom_pre_pass(graph)
    count += 1
```