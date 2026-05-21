import torch
import tensorflow as tf
import shutil
import os

# Clean up previous logs to ensure a clean test environment
log_dir = "/tmp/tf_summary_graph_test"
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)

# Define the function mimicking the PyTorch logic from the bug report.
# This function performs a series of shape manipulations resulting in a scalar (0-d tensor).
@tf.function
def fuzzed_program(sentinel):
    # var_node_3 = torch.full((2, 3), 3, dtype=torch.int32)
    var_node_3 = tf.fill([2, 3], tf.constant(3, dtype=tf.int32))

    # _inp_unique_wide = torch.arange(1, device=var_node_3.device, dtype=torch.int64)
    _inp_unique_wide = tf.range(1, dtype=tf.int64)

    # _uniq_wide = torch.unique(_inp_unique_wide)
    # TF unique returns (values, indices), we only need values
    _uniq_wide, _ = tf.unique(_inp_unique_wide)

    # var_node_2 = _uniq_wide.to(var_node_3.dtype)
    var_node_2 = tf.cast(_uniq_wide, var_node_3.dtype)

    # var_node_1 = torch.reshape(var_node_2, [1])
    var_node_1 = tf.reshape(var_node_2, [1])

    # var_node_0 = torch.squeeze(var_node_1)
    # This operation reduces the dimensionality to 0 (scalar), which is the 
    # critical part of the original bug regarding dimensionality of sizes/strides.
    var_node_0 = tf.squeeze(var_node_1)

    # result = var_node_0 * sentinel
    result = var_node_0 * sentinel

    # Handle complex numbers if necessary
    if result.dtype.is_complex:
        result = tf.math.real(result)

    return result

# Sentinel tensor
sentinel = tf.constant(1.0)

# 1. Test eager execution (equivalent to PyTorch eager)
print("Testing eager execution...")
try:
    result_eager = fuzzed_program(sentinel)
    print(f" eager success: {result_eager.numpy()}")
except Exception as e:
    print(f" eager failed: {e}")
    raise

# 2. Test tf.summary.graph (the similar API)
# The original bug involved a divergence between eager and compiled modes.
# In TensorFlow, tf.summary.graph operates on the graph structure (compiled representation).
# We test if tf.summary.graph can correctly serialize the graph containing the 
# scalar-producing operations from the bug report.
print("Testing tf.summary.graph...")
try:
    writer = tf.summary.create_file_writer(log_dir)
    with writer.as_default():
        # Get the concrete function to access the graph
        concrete_func = fuzzed_program.get_concrete_function(sentinel)
        
        # Call the similar API: tf.summary.graph
        # This tests the API's ability to handle the specific graph topology derived from the bug.
        tf.summary.graph(concrete_func.graph)
        
        print(" tf.summary.graph success")
        
        # Verify that the summary was written
        writer.flush()
        assert os.path.exists(log_dir), "Log directory was not created"
        files = os.listdir(log_dir)
        assert len(files) > 0, "No event files were written"
        
except Exception as e:
    print(f" tf.summary.graph failed: {e}")
    raise
finally:
    # Clean up
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)