import torch
import tensorflow as tf
import numpy as np

# Set seeds for reproducibility
tf.random.set_seed(13653)
np.random.seed(13653)

def fuzzed_program(arg_0, arg_1, sentinel):
    # Adapt the logic to use tf.keras.name_scope
    with tf.keras.name_scope("fuzzed_logic"):
        # var_node_3 = torch.full((), 1.0, dtype=torch.float32)
        var_node_3 = tf.constant(1.0, dtype=tf.float32)
        
        # var_node_2 = var_node_3.item() -> Keep as 0-d tensor in TF
        var_node_2 = var_node_3
        
        # var_node_5 = -3 (dtype=int32)
        var_node_5 = tf.constant(-3, dtype=tf.int32)
        
        # var_node_6 = arg_0 (dtype=int64)
        var_node_6 = tf.cast(arg_0, tf.int64)
        
        # var_node_4 = var_node_5 + var_node_6 (dtype=int32)
        # Explicit cast required in TF for int32 + int64
        var_node_4 = var_node_5 + tf.cast(var_node_6, tf.int32)
        
        # var_node_1 = var_node_2 + var_node_4 (dtype=float32)
        var_node_1 = var_node_2 + tf.cast(var_node_4, tf.float32)
        
        # var_node_9 = 1 (dtype=int64)
        var_node_9 = tf.constant(1, dtype=tf.int64)
        
        # var_node_10 = -10 (dtype=int32)
        var_node_10 = tf.constant(-10, dtype=tf.int32)
        
        # var_node_8 = var_node_9 / var_node_10 (dtype=int64)
        # Using floordiv to preserve integer type as implied by the bug report comments
        var_node_8 = tf.math.floordiv(var_node_9, tf.cast(var_node_10, tf.int64))
        
        # var_node_12 = arg_1 (dtype=int64)
        var_node_12 = tf.cast(arg_1, tf.int64)
        
        # var_node_13 = -5 (dtype=int32)
        var_node_13 = tf.constant(-5, dtype=tf.int32)
        
        # var_node_11 = var_node_12 / var_node_13 (dtype=int32)
        var_node_11 = tf.math.floordiv(tf.cast(var_node_12, tf.int32), var_node_13)
        
        # var_node_7 = var_node_8 + var_node_11 (dtype=int32)
        var_node_7 = tf.cast(var_node_8, tf.int32) + var_node_11
        
        # var_node_0 = var_node_1 * var_node_7 (dtype=float32)
        var_node_0 = var_node_1 * tf.cast(var_node_7, tf.float32)
        
        # result = var_node_0 * sentinel
        result = var_node_0 * sentinel
        
        # Check for complex type
        if result.dtype in (tf.complex64, tf.complex128):
            result = tf.math.real(result)
            
        return result

# Generate inputs similar to the original PyTorch code
# arg_0 = torch.tensor(torch.randn(()), dtype=torch.int64).item()
# Simulating randn() -> cast to int64 -> item()
arg_0 = int(np.random.randn())
arg_1 = int(np.random.randn())

# Sentinel tensor to ensure gradient computation
sentinel = tf.Variable(1.0, dtype=tf.float32)

# 1. Test Eager Execution
print("Testing Eager Execution...")
with tf.GradientTape() as tape:
    result_eager = fuzzed_program(arg_0, arg_1, sentinel)
print(f' eager success: {result_eager.numpy()}')

# 2. Test Compiled Execution (tf.function)
# This mimics the behavior of torch.compile to check for divergence
print("Testing Compiled Execution (tf.function)...")
compiled_program = tf.function(fuzzed_program)

with tf.GradientTape() as tape:
    result_compiled = compiled_program(arg_0, arg_1, sentinel)
print(f' compile success: {result_compiled.numpy()}')

# Verify consistency
assert np.allclose(result_eager.numpy(), result_compiled.numpy()), \
    f"Divergence detected: Eager={result_eager.numpy()}, Compiled={result_compiled.numpy()}"
print(" No divergence between eager and compiled modes.")