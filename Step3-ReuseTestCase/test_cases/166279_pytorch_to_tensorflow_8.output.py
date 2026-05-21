import torch
import tensorflow as tf
import numpy as np

# Set seeds for reproducibility
tf.random.set_seed(1166094474)
np.random.seed(1166094474)

# Sentinel tensor to ensure gradient computation (mimicking PyTorch requires_grad)
sentinel = tf.Variable(1.0, dtype=tf.float32)

# Input tensors adapted from PyTorch test case
# arg_0: size=(12,), dtype=bool
arg_0 = tf.cast(tf.random.uniform((12,), 0, 2, dtype=tf.int32), tf.bool)
# arg_1: size=(10,), dtype=int64 (Unused in original logic flow but kept for signature)
arg_1 = tf.cast(tf.random.uniform((10,), 5, 30, dtype=tf.int32), tf.int64)
# arg_2: size=(6, 4), dtype=bool
arg_2 = tf.cast(tf.random.uniform((6, 4), 0, 2, dtype=tf.int32), tf.bool)
# arg_3: size=(2,), dtype=bool
arg_3 = tf.cast(tf.random.uniform((2,), 0, 2, dtype=tf.int32), tf.bool)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, sentinel):
    # Replicate logic: var_node_3 = torch.full((12,), False, dtype=torch.bool)
    var_node_3 = tf.fill((12,), False)
    
    # Replicate logic: var_node_2 = torch.chunk(var_node_3, 4, dim=0)[0]
    # PyTorch chunk splits into roughly equal parts. tf.split requires exact sizes or num_splits.
    # 12 / 4 = 3. So we split into 4 parts of size 3.
    var_node_2 = tf.split(var_node_3, 4, axis=0)[0] # size=(3,)

    # Replicate logic: var_node_6 = arg_0
    var_node_6 = arg_0
    
    # Replicate logic: var_node_5 = torch.gather(var_node_6, 0, _index_var_node_5)
    _input_size_var_node_5 = tf.shape(var_node_6)[0]
    _index_var_node_5 = tf.random.uniform((10,), 0, _input_size_var_node_5, dtype=tf.int32)
    var_node_5 = tf.gather(var_node_6, _index_var_node_5) # size=(10,)

    # Replicate logic: var_node_4 = torch.chunk(var_node_5, 2, dim=0)[0]
    # 10 / 2 = 5.
    var_node_4 = tf.split(var_node_5, 2, axis=0)[0] # size=(5,)

    # --- Adaptation to tf.keras.ops.triu ---
    # Original: var_node_10 = arg_2
    var_node_10 = arg_2 # size=(6, 4)
    
    # Original: var_node_9 = torch.chunk(var_node_10, 4, dim=1)[0]
    # Adaptation: Apply tf.keras.ops.triu instead of chunk.
    # triu returns a tensor of the same shape (6, 4).
    var_node_9 = tf.keras.ops.triu(var_node_10) 

    # Original: var_node_8 = torch.squeeze(var_node_9)
    # Original var_node_9 was (6, 1), squeezing to (6,).
    # Current var_node_9 is (6, 4). Squeezing does nothing unless we reshape.
    # To maintain the concatenation logic (which expects 1D tensors), we flatten.
    var_node_8 = tf.reshape(var_node_9, [-1]) # size=(24,)

    # Replicate logic: var_node_1 = torch.cat([var_node_2, var_node_4, var_node_8], dim=0)
    # var_node_2 (3,) + var_node_4 (5,) + var_node_8 (24,) = (32,)
    var_node_1 = tf.concat([var_node_2, var_node_4, var_node_8], axis=0)

    # Replicate logic: var_node_11 = arg_3
    var_node_11 = arg_3
    
    # Replicate logic: var_node_0 = torch.cat([var_node_1, var_node_11], dim=0)
    # var_node_1 (32,) + var_node_11 (2,) = (34,)
    var_node_0 = tf.concat([var_node_1, var_node_11], axis=0)

    # Ensure gradient computation by multiplying with sentinel
    result = var_node_0 * sentinel
    return result

# Run Eager
print("Running Eager...")
try:
    result_eager = fuzzed_program(arg_0, arg_1, arg_2, arg_3, sentinel)
    print(" eager success")
except Exception as e:
    print(f" eager failed: {e}")

# Run Compiled (tf.function with JIT)
print("Running Compiled...")
compiled_program = tf.function(fuzzed_program, jit_compile=True)
try:
    result_compiled = compiled_program(arg_0, arg_1, arg_2, arg_3, sentinel)
    print(" compile success")
except Exception as e:
    print(f" compile failed: {e}")

# Check for divergence
if 'result_eager' in locals() and 'result_compiled' in locals():
    if not tf.reduce_all(tf.equal(result_eager, result_compiled)):
        print(" Divergence detected between eager and compiled!")
        print(f"Eager result: {result_eager}")
        print(f"Compiled result: {result_compiled}")
    else:
        print(" No divergence detected")