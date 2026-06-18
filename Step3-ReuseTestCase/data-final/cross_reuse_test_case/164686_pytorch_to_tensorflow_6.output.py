import torch
import tensorflow as tf
import numpy as np

# Set seeds for reproducibility
tf.random.set_seed(13653)
np.random.seed(13653)

# Define the function to be tested, wrapped in tf.function to simulate compilation
# and using tf.compat.v1.name_scope as requested.
@tf.function
def fuzzed_program(arg_0, arg_1, sentinel):
    # Use the specific API requested: tf.compat.v1.name_scope
    with tf.compat.v1.name_scope("fuzzed_logic"):
        # var_node_3 = torch.full((), 1.0, dtype=torch.float32)
        var_node_3 = tf.constant(1.0, dtype=tf.float32)
        
        # var_node_2 = var_node_3.item()
        # In graph mode, we keep it as a tensor to maintain the graph structure
        var_node_2 = var_node_3
        
        # var_node_5 = -3 (dtype=int32)
        var_node_5 = tf.constant(-3, dtype=tf.int32)
        
        # var_node_6 = arg_0 (dtype=int64)
        # Cast input to int64 to match the logic
        var_node_6 = tf.cast(arg_0, dtype=tf.int64)
        
        # var_node_4 = var_node_5 + var_node_6 (dtype=int32 + int64)
        # TensorFlow promotes int32 + int64 to int64
        var_node_4 = tf.cast(var_node_5, dtype=tf.int64) + var_node_6
        
        # var_node_1 = var_node_2 + var_node_4 (float32 + int64 -> float32)
        var_node_1 = var_node_2 + tf.cast(var_node_4, dtype=tf.float32)
        
        # var_node_9 = 1 (dtype=int64)
        var_node_9 = tf.constant(1, dtype=tf.int64)
        
        # var_node_10 = -10 (dtype=int32)
        var_node_10 = tf.constant(-10, dtype=tf.int32)
        
        # var_node_8 = var_node_9 / var_node_10
        # Division of integers in TF (using /) results in float.
        # We cast to float to perform true division as implied by the operator.
        var_node_8 = tf.cast(var_node_9, dtype=tf.float32) / tf.cast(var_node_10, dtype=tf.float32)
        
        # var_node_12 = arg_1 (dtype=int64)
        var_node_12 = tf.cast(arg_1, dtype=tf.int64)
        
        # var_node_13 = -5 (dtype=int32)
        var_node_13 = tf.constant(-5, dtype=tf.int32)
        
        # var_node_11 = var_node_12 / var_node_13
        var_node_11 = tf.cast(var_node_12, dtype=tf.float32) / tf.cast(var_node_13, dtype=tf.float32)
        
        # var_node_7 = var_node_8 + var_node_11
        var_node_7 = var_node_8 + var_node_11
        
        # var_node_0 = var_node_1 * var_node_7
        var_node_0 = var_node_1 * var_node_7
        
        # result = var_node_0 * sentinel
        result = var_node_0 * sentinel
        
        # Check for complex type and take real part if necessary
        if result.dtype == tf.complex64 or result.dtype == tf.complex128:
            result = tf.math.real(result)
            
        return result

# Prepare inputs
# arg_0 = torch.tensor(torch.randn(()), dtype=torch.int64).item()
# Generates a random float, casts to int64, then extracts as Python int.
arg_0 = int(np.random.randn())
arg_1 = int(np.random.randn())

# Sentinel tensor
# sentinel = torch.tensor(1.0, requires_grad=True)
sentinel = tf.constant(1.0, dtype=tf.float32)

# Execute the test
try:
    # We call the function. Since it is decorated with @tf.function,
    # this triggers the tracing/compilation process in TensorFlow.
    result_tf = fuzzed_program(arg_0, arg_1, sentinel)
    
    # Force execution to ensure no runtime errors
    print(f"Result: {result_tf.numpy()}")
    print(" TensorFlow execution success")

except Exception as e:
    print(f" TensorFlow execution failed: {e}")