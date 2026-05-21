import torch
import tensorflow as tf
import numpy as np

# Replicating the structure of the original PyTorch test case
# adapted for TensorFlow and the tf.keras.activations.relu API.

def fuzzed_program(arg_0, arg_1):
    # var_node_3 = arg_0 # size=(2, 10), dtype=float64
    # var_node_4 = arg_1 # size=(10, 3), dtype=float64
    
    # Matmul operation
    var_node_2 = tf.matmul(arg_0, arg_1) # size=(2, 3), dtype=float64
    
    # Original PyTorch code used torch.unique on arange(1).
    # We adapt this to use tf.keras.activations.relu.
    # Input generation: tf.range(1) -> [0]
    _inp_relu_wide = tf.range(1, dtype=tf.int64)
    
    # Cast to float64 for ReLU and subsequent matmul
    _inp_relu_float = tf.cast(_inp_relu_wide, tf.float64)
    
    # Apply the similar API: tf.keras.activations.relu
    # ReLU([0.0]) -> [0.0]
    _relu_wide = tf.keras.activations.relu(_inp_relu_float)
    
    var_node_1 = _relu_wide # size=(1,), dtype=float64
    
    # Create the full tensor
    var_node_5 = tf.fill([1, 18], tf.cast(0.40330381448978797, tf.float64)) # size=(1, 18), dtype=float64
    
    # Matmul operation where the original bug occurred (shape mismatch)
    # var_node_1 is (1,), var_node_5 is (1, 18)
    var_node_0 = tf.matmul(var_node_1, var_node_5) # Expected size=(18,)
    
    return var_node_0

# Setup inputs matching the original shapes and dtypes
# arg_0: size=(2, 10), dtype=float64
arg_0 = tf.random.normal((2, 10), dtype=tf.float64)
# arg_1: size=(10, 3), dtype=float64
arg_1 = tf.random.normal((10, 3), dtype=tf.float64)

print("Testing Eager Execution...")
try:
    result_eager = fuzzed_program(arg_0, arg_1)
    print(f" Eager success. Result shape: {result_eager.shape}")
except Exception as e:
    print(f" Eager failed: {e}")

print("\nTesting Compiled Execution (tf.function)...")
try:
    compiled_program = tf.function(fuzzed_program)
    result_compiled = compiled_program(arg_0, arg_1)
    print(f" Compile success. Result shape: {result_compiled.shape}")
    
    # Verify consistency
    if result_eager.shape == result_compiled.shape:
        print(" Shape consistency check passed.")
    else:
        print(f" Shape divergence: Eager {result_eager.shape} vs Compiled {result_compiled.shape}")
        
except Exception as e:
    print(f" Compile failed: {e}")