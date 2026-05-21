import torch
import tensorflow as tf
import numpy as np

# Replicate the configuration and setup from the PyTorch bug report
# Note: TensorFlow does not have a direct equivalent to torch._dynamo.config,
# but we ensure the logic is preserved.

tf.random.set_seed(19989)

# Sentinel tensor to ensure gradient computation
# PyTorch: torch.tensor(1.0, requires_grad=True)
# TensorFlow: tf.Variable(1.0)
sentinel = tf.Variable(1.0)

# Argument 0
# PyTorch: torch.tensor(torch.randn(()), dtype=torch.int32).item()
# This generates a random float, casts to int32, and extracts a Python scalar.
arg_0 = tf.cast(tf.random.normal(()), tf.int32).numpy().item()

def fuzzed_program(arg_0, sentinel):
    # var_node_2 = -6 # dtype=int64
    # We use a tensor to ensure type consistency in the graph
    var_node_2 = tf.constant(-6, dtype=tf.int64)
    
    # var_node_3 = arg_0 # dtype=int32
    # Cast input to int32 tensor to match the type annotation in the bug report
    var_node_3 = tf.constant(arg_0, dtype=tf.int32)
    
    # var_node_1 = var_node_2 * var_node_3 # dtype=int32
    # PyTorch comment says result is int32. 
    # In TF, int64 * int32 is invalid, so we cast var_node_2 to int32 to match the expected output type.
    var_node_1 = tf.cast(var_node_2, dtype=tf.int32) * var_node_3
    
    # var_node_5 = torch.full((), 1, dtype=torch.int64)
    var_node_5 = tf.constant(1, dtype=tf.int64)
    
    # var_node_4 = var_node_5.item() # dtype=int64
    # Extracting a Python scalar from the tensor
    var_node_4 = var_node_5.numpy().item()
    
    # var_node_0 = var_node_1 / var_node_4 # dtype=int64 (per comment)
    # Note: Standard division (/) in PyTorch/TF usually results in float.
    # The bug report comment says int64, implying floor division or specific casting behavior.
    # We use standard division here to match the operator '/', but cast to int64 if that was the strict intent.
    # Given the divergence bug, we stick to the operator '/'.
    var_node_0 = var_node_1 / var_node_4
    
    # Ensure gradient computation by multiplying with sentinel
    result = var_node_0 * sentinel
    
    return result

# 1. Run Eager
print("Running Eager...")
try:
    result_original = fuzzed_program(arg_0, sentinel)
    print(' eager success')
except Exception as e:
    print(f' eager failed: {e}')

# 2. Run Compiled (tf.function) with name_scope
# The prompt asks to adapt to tf.name_scope.
# We wrap the logic in tf.name_scope inside a tf.function to simulate the compilation context.
print("Running Compiled with name_scope...")
@tf.function
def compiled_program(arg_0, sentinel):
    with tf.name_scope("fuzzed_program_scope"):
        return fuzzed_program(arg_0, sentinel)

try:
    result_compiled = compiled_program(arg_0, sentinel)
    print(' compile success')
except Exception as e:
    print(f' compile failed: {e}')

# Optional: Check for divergence if both succeeded
if 'result_original' in locals() and 'result_compiled' in locals():
    if np.allclose(result_original.numpy(), result_compiled.numpy()):
        print(" Results match")
    else:
        print(" Results diverge")