import torch
import numpy as np
import sys

# Handle environment dependency issues (e.g., GLIBC version mismatch)
try:
    import tensorflow as tf
except ImportError as e:
    print(f"Skipping test: Failed to import TensorFlow due to environment dependency issues.")
    print(f"Error details: {e}")
    sys.exit(0)

# Setup reproducibility
tf.random.set_seed(114503)

def fuzzed_program(arg_0, arg_1, sentinel):
    # Replicate tensor creation logic
    # var_node_4 = torch.full((2, 3), 3, dtype=torch.int16)
    var_node_4 = tf.fill((2, 3), tf.cast(3, tf.int16))
    
    # var_node_3 = torch.unique(var_node_4)
    # tf.unique requires 1D input, so we flatten to match the intent
    var_node_4_flat = tf.reshape(var_node_4, [-1])
    var_node_3 = tf.unique(var_node_4_flat)[0]
    
    # var_node_2 = torch.squeeze(var_node_3)
    var_node_2 = tf.squeeze(var_node_3)
    
    # var_node_7 = arg_0
    var_node_7 = arg_0
    # var_node_8 = arg_1
    var_node_8 = arg_1
    
    # var_node_6 = torch.sub(var_node_7, var_node_8)
    var_node_6 = tf.subtract(var_node_7, var_node_8)
    
    # var_node_10 = torch.full((1,), 3, dtype=torch.int16)
    var_node_10 = tf.fill((1,), tf.cast(3, tf.int16))
    
    # var_node_9 = torch.squeeze(var_node_10)
    var_node_9 = tf.squeeze(var_node_10)
    
    # var_node_5 = torch.add(var_node_6, var_node_9)
    var_node_5 = tf.add(var_node_6, var_node_9)
    
    # --- Leveraging Similar API ---
    # Original: var_node_1 = torch.div(var_node_2, var_node_5)
    # Using tf.experimental.numpy.exp to test type handling (int16 -> float promotion)
    # This replaces the division operation with the exponential operation from the similar API
    var_node_1 = tf.experimental.numpy.exp(var_node_5)
    
    # Original: var_node_0 = var_node_1.item()
    # In TensorFlow, 0-d tensors act as scalars in operations. 
    # We use the tensor directly to preserve the graph structure for compilation.
    var_node_0 = var_node_1
    
    # Original: result = var_node_0 * sentinel
    # Ensure gradient computation by multiplying with sentinel
    result = var_node_0 * sentinel
    
    return result

# Sentinel tensor to ensure gradient computation
sentinel = tf.constant(1.0)

# Create arguments similar to the original test case
# torch.as_strided(..., (), ()) creates a 0-d tensor (scalar)
arg_0 = tf.constant(10, dtype=tf.int16)
arg_1 = tf.constant(5, dtype=tf.int16)

# Test Eager
print("Testing Eager...")
try:
    result_eager = fuzzed_program(arg_0, arg_1, sentinel)
    print(f" eager success: {result_eager.numpy()}")
except Exception as e:
    print(f" eager failed: {e}")

# Test Compiled (tf.function)
print("Testing Compiled...")
compiled_program = tf.function(fuzzed_program)
try:
    result_compiled = compiled_program(arg_0, arg_1, sentinel)
    print(f" compile success: {result_compiled.numpy()}")
except Exception as e:
    print(f" compile failed: {e}")