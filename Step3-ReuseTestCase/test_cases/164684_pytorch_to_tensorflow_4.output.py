import torch
import tensorflow as tf

# Ensure we are in TF2 eager mode by default
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()

def fuzzed_program(arg_0, sentinel):
    # arg_0 is a string tensor, mimicking the boolean tensor from the original bug
    # size=(1,), dtype=string
    var_node_1 = tf.squeeze(arg_0) # size=(), dtype=string
    
    # Attempt to extract the scalar value similar to .item() in PyTorch
    # In eager mode, .numpy() works to extract the value.
    # In graph mode (tf.function), this will fail because .numpy() is not available on symbolic tensors.
    var_node_0 = var_node_1.numpy()
    
    # Call the target API: tf.compat.v1.resource_loader.readahead_file_path
    # This function expects a string path and returns it.
    result = tf.compat.v1.resource_loader.readahead_file_path(var_node_0, sentinel)
    return result

# Sentinel argument (readahead size)
sentinel = "128M"

# Input argument: a string tensor that will be squeezed to a scalar
arg_0 = tf.constant(["/tmp/model_data"])

# Test Eager Mode
try:
    result_original = fuzzed_program(arg_0, sentinel)
    print(f' eager success: {result_original}')
except Exception as e:
    print(f' eager failed: {e}')

# Test Compiled Mode (tf.function)
# This mimics torch.compile. We expect a divergence here because .numpy() 
# cannot be called on a symbolic tensor inside a graph.
compiled_program = tf.function(fuzzed_program)
try:
    result_compiled = compiled_program(arg_0, sentinel)
    print(f' compile success: {result_compiled}')
except Exception as e:
    # Expected to fail in compile mode due to .numpy() call on symbolic tensor
    print(f' compile failed: {e}')