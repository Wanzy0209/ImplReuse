import torch
import tensorflow as tf
import numpy as np

# This test case mirrors the structure of the original PyTorch bug report.
# It replaces torch._dynamo.functional_export._dynamo_graph_capture_for_export 
# with TensorFlow's tf.function for graph capture.
# It replaces torch.nn.attention.flex_attention with the similar API 
# tf.keras.backend.ctc_label_dense_to_sparse.

class CTCModule(tf.Module):
    """
    A module wrapping the similar API (ctc_label_dense_to_sparse),
    analogous to FlexAttentionModule in the original issue.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, labels, label_lengths=None):
        # Using the similar API: tf.keras.backend.ctc_label_dense_to_sparse
        # This mirrors the usage of flex_attention with block_mask kwarg.
        return tf.keras.backend.ctc_label_dense_to_sparse(labels, label_lengths)

# Instantiate the model
ctc_model = CTCModule()

# Setup inputs
batch_size = 2
max_num_labels = 5
num_classes = 10

# Create dense labels (analogous to query/key/value tensors)
labels = tf.constant(
    np.random.randint(0, num_classes, size=(batch_size, max_num_labels)), 
    dtype=tf.int32
)
# Create label lengths (analogous to block_mask - a structural auxiliary input)
label_lengths = tf.constant([3, 4], dtype=tf.int32)

# Define inputs and kwargs to match the original pattern
ctc_inputs = (labels,)
ctc_kwargs = {"label_lengths": label_lengths}

# 1. Run Eagerly (Baseline)
eager_out = ctc_model(*ctc_inputs, **ctc_kwargs)
print("Eager execution successful.")

# 2. Run with Graph Capture (Analogous to _dynamo_graph_capture_for_export)
# In TensorFlow, tf.function is the standard mechanism for graph capture/tracing.
try:
    # We trace the module, similar to how _dynamo_graph_capture_for_export traces the PyTorch model
    traced_model = tf.function(ctc_model)
    graph_out = traced_model(*ctc_inputs, **ctc_kwargs)
    
    # Verify that the graph output matches the eager output
    # Comparing SparseTensors requires checking indices, values, and shape
    assert tf.reduce_all(tf.equal(eager_out.indices, graph_out.indices)), "Indices mismatch"
    assert tf.reduce_all(tf.equal(eager_out.values, graph_out.values)), "Values mismatch"
    assert tf.reduce_all(tf.equal(eager_out.dense_shape, graph_out.dense_shape)), "Shape mismatch"
    
    print("Test Passed: Graph capture successful for ctc_label_dense_to_sparse with kwarg.")

except Exception as e:
    print(f"Test Failed: Graph capture failed with error: {e}")