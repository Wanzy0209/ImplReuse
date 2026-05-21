import torch
import tensorflow as tf
import numpy as np

# Setup seed to match the original test context
tf.random.set_seed(9)

def fuzzed_program(arg_0):
    """
    Adapted from the PyTorch fuzzed program to test tf.keras.backend.random_normal.
    The original bug involved torch.nonzero inside a compiled graph causing stride issues.
    Here we verify that random_normal behaves consistently in eager vs compiled modes.
    """
    # var_node_1 = arg_0 # size=(1, 2), dtype=int64
    var_node_1 = arg_0
    
    # var_node_5 = torch.full((1, 2), -66, dtype=torch.int32)
    var_node_5 = tf.fill((1, 2), -66)
    var_node_5 = tf.cast(var_node_5, tf.int32)
    
    # var_node_6 = torch.full((1, 2), 77, dtype=torch.int64)
    var_node_6 = tf.fill((1, 2), 77)
    var_node_6 = tf.cast(var_node_6, tf.int64)
    
    # var_node_4 = torch.ops.aten.add(var_node_5, var_node_6)
    # PyTorch result type is int32
    var_node_4 = tf.cast(var_node_5, tf.int32) + tf.cast(var_node_6, tf.int32)
    
    # var_node_7 = torch.full((1, 2), -64, dtype=torch.int32)
    var_node_7 = tf.fill((1, 2), -64)
    var_node_7 = tf.cast(var_node_7, tf.int32)
    
    # var_node_3 = torch.ops.aten.mul(var_node_4, var_node_7)
    var_node_3 = var_node_4 * var_node_7
    
    # var_node_9 = torch.full((3, 4), False, dtype=torch.bool)
    var_node_9 = tf.fill((3, 4), False)
    
    # --- API SUBSTITUTION ---
    # Original: var_node_8 = torch.nonzero(var_node_9)
    # Similar API: tf.keras.backend.random_normal
    # Note: random_normal returns float32. We use shape (1, 2) to maintain compatibility 
    # with subsequent operations (var_node_3 is (1, 2)).
    var_node_8 = tf.keras.backend.random_normal(shape=(1, 2), mean=0.0, stddev=1.0)
    
    # var_node_2 = torch.ops.aten.add(var_node_3, var_node_8)
    # Type adaptation: var_node_3 is int32, var_node_8 is float32. Cast to float32.
    var_node_2 = tf.cast(var_node_3, tf.float32) + var_node_8
    
    # var_node_0 = torch.ops.aten.div(var_node_1, var_node_2)
    # Type adaptation: var_node_1 is int64, var_node_2 is float32. Cast to float32.
    var_node_0 = tf.cast(var_node_1, tf.float32) / var_node_2
    
    return var_node_0

# Input initialization
# arg_0 = torch.randint(0, 3, (1, 2), dtype=torch.int64)
arg_0 = tf.constant(np.random.randint(0, 3, (1, 2)), dtype=tf.int64)

# 1. Test Eager Execution
print("Testing Eager Mode...")
with tf.GradientTape() as tape:
    tape.watch(arg_0)
    result_eager = fuzzed_program(arg_0)
grads_eager = tape.gradient(result_eager, arg_0)
print(f" Eager success. Result shape: {result_eager.shape}, dtype: {result_eager.dtype}")

# 2. Test Compiled Execution (tf.function)
print("\nTesting Compiled Mode (tf.function)...")
compiled_program = tf.function(fuzzed_program)

with tf.GradientTape() as tape:
    tape.watch(arg_0)
    result_compiled = compiled_program(arg_0)
grads_compiled = tape.gradient(result_compiled, arg_0)
print(f" Compile success. Result shape: {result_compiled.shape}, dtype: {result_compiled.dtype}")

# 3. Verification
# Check shape and dtype consistency
assert result_eager.shape == result_compiled.shape, \
    f"Shape mismatch: Eager {result_eager.shape} vs Compiled {result_compiled.shape}"
assert result_eager.dtype == result_compiled.dtype, \
    f"Dtype mismatch: Eager {result_eager.dtype} vs Compiled {result_compiled.dtype}"

print(f"\nEager result sample: {result_eager.numpy()}")
print(f"Compiled result sample: {result_compiled.numpy()}")

# Note: We do not assert exact value equality for random_normal outputs 
# because random operations may be re-sampled during tracing or execution 
# depending on the specific backend implementation, but the structural 
# consistency (shape/dtype) is verified.