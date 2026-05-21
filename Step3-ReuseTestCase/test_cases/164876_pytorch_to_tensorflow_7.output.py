import torch
import tensorflow as tf
import numpy as np

# Set seeds for reproducibility
tf.random.set_seed(1012969)
np.random.seed(1012969)

def fuzzed_program(arg_0, arg_1):
    # var_node_3 = arg_0 # size=(2, 10), dtype=float64
    # var_node_4 = arg_1 # size=(10, 3), dtype=float64
    # var_node_2 = torch.matmul(var_node_3, var_node_4) # size=(2, 3)
    var_node_3 = arg_0
    var_node_4 = arg_1
    var_node_2 = tf.matmul(tf.cast(var_node_3, tf.float64), tf.cast(var_node_4, tf.float64))

    # Original: _inp_unique_wide = torch.arange(1, ...); _uniq_wide = torch.unique(_inp_unique_wide)
    # Adaptation: Use tf.keras.ops.add to mimic the operation producing a tensor of size (1,)
    # We create a tensor of size 1 and add it to another tensor of size 1.
    _inp_add = tf.constant([0], dtype=tf.int64)
    # Using the similar API: tf.keras.ops.add
    _added = tf.keras.ops.add(_inp_add, tf.constant([0], dtype=tf.int64))
    
    # var_node_1 = _uniq_wide.to(var_node_2.dtype) # size=(1,), dtype=float64
    var_node_1 = tf.cast(_added, tf.float64)

    # var_node_5 = torch.full((1, 18), 0.4033..., dtype=torch.float64) # size=(1, 18)
    var_node_5 = tf.fill((1, 18), tf.constant(0.40330381448978797, dtype=tf.float64))

    # var_node_0 = torch.matmul(var_node_1, var_node_5) # size=(18,)
    # (1,) x (1, 18) -> (18,)
    var_node_0 = tf.matmul(tf.cast(var_node_1, tf.float64), tf.cast(var_node_5, tf.float64))

    return var_node_0

# Create inputs
# arg_0 = torch.as_strided(..., (2, 10), ...)
# arg_1 = torch.as_strided(..., (10, 3), ...)
arg_0 = tf.constant(np.random.randn(2, 10), dtype=tf.float64)
arg_1 = tf.constant(np.random.randn(10, 3), dtype=tf.float64)

args = (arg_0, arg_1)

# Run Eager
print('Running eager...')
result_original = fuzzed_program(*args)
print(f' eager success, shape: {result_original.shape}')

# Run Compiled (using tf.function with XLA to mimic torch.compile strictness)
print('Running compiled...')
compiled_program = tf.function(fuzzed_program, jit_compile=True)
result_compiled = compiled_program(*args)
print(f' compile success, shape: {result_compiled.shape}')

# Verify consistency
if tf.reduce_all(tf.equal(result_original, result_compiled)):
    print(' Results match between eager and compiled modes.')
else:
    print(' Divergence detected between eager and compiled modes.')