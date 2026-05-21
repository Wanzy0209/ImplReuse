import torch
import tensorflow as tf
import numpy as np

# Set seed for reproducibility, matching the original fuzzer seed
tf.random.set_seed(1166094474)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, sentinel):
    # var_node_3 = torch.full((12,), False, dtype=torch.bool)
    var_node_3 = tf.fill((12,), False)

    # var_node_2 = torch.chunk(var_node_3, 4, dim=0)[0]
    # Adaptation: Use tf.experimental.numpy.swapaxes.
    # Note: swapaxes on a 1D tensor is a no-op, but we call it to match the API usage pattern.
    var_node_2 = tf.experimental.numpy.swapaxes(var_node_3, 0, 0)

    # var_node_6 = arg_0
    var_node_6 = arg_0

    # var_node_7 = arg_1
    var_node_7 = arg_1

    # _input_size_var_node_5 = var_node_6.size(0)
    _input_size_var_node_5 = tf.shape(var_node_6)[0]

    # _index_var_node_5 = torch.randint(0, _input_size_var_node_5, (10,), device=var_node_6.device)
    _index_var_node_5 = tf.random.uniform((10,), minval=0, maxval=_input_size_var_node_5, dtype=tf.int32)

    # var_node_5 = torch.gather(var_node_6, 0, _index_var_node_5)
    var_node_5 = tf.gather(var_node_6, _index_var_node_5)

    # var_node_4 = torch.chunk(var_node_5, 2, dim=0)[0]
    # Adaptation: Use tf.experimental.numpy.swapaxes (no-op on 1D).
    var_node_4 = tf.experimental.numpy.swapaxes(var_node_5, 0, 0)

    # var_node_10 = arg_2
    var_node_10 = arg_2

    # var_node_9 = torch.chunk(var_node_10, 4, dim=1)[0]
    # Adaptation: Use tf.experimental.numpy.swapaxes.
    # Original shape (6, 4). Swapping axes 0 and 1 results in shape (4, 6).
    var_node_9 = tf.experimental.numpy.swapaxes(var_node_10, 0, 1)

    # var_node_8 = torch.squeeze(var_node_9)
    # Original logic: (6, 1) -> (6,).
    # New logic: (4, 6). Squeeze does nothing as no dimension is 1.
    # To maintain the program flow (concatenation), we reshape to 1D.
    var_node_8 = tf.reshape(var_node_9, [-1])

    # var_node_1 = torch.cat([var_node_2, var_node_4, var_node_8], dim=0)
    var_node_1 = tf.concat([var_node_2, var_node_4, var_node_8], 0)

    # var_node_11 = arg_3
    var_node_11 = arg_3

    # var_node_0 = torch.cat([var_node_1, var_node_11], dim=0)
    var_node_0 = tf.concat([var_node_1, var_node_11], 0)

    # result = var_node_0 * sentinel
    # Cast bool to float for multiplication
    result = tf.cast(var_node_0, tf.float32) * sentinel
    return result

# Prepare inputs mimicking the original fuzzer's tensor properties
# arg_0: size=(12,), stride=(1,), dtype=bool
arg_0 = tf.cast(tf.random.uniform((12,), 0, 2, dtype=tf.int32), tf.bool)

# arg_1: size=(10,), stride=(1,), dtype=int64
arg_1 = tf.cast(tf.random.uniform((10,), 5, 30, dtype=tf.int32), tf.int64)

# arg_2: size=(6, 4), stride=(4, 1), dtype=bool
arg_2 = tf.cast(tf.random.uniform((6, 4), 0, 2, dtype=tf.int32), tf.bool)

# arg_3: size=(2,), stride=(1,), dtype=bool
arg_3 = tf.cast(tf.random.uniform((2,), 0, 2, dtype=tf.int32), tf.bool)

# Sentinel tensor to ensure gradient computation (or graph connection)
sentinel = tf.constant(1.0)

# Run Eager
print('Running eager execution...')
try:
    result_eager = fuzzed_program(arg_0, arg_1, arg_2, arg_3, sentinel)
    print(' eager success')
except Exception as e:
    print(f' eager failed: {e}')

# Run Compiled
print('Running compiled execution (tf.function)...')
try:
    compiled_program = tf.function(fuzzed_program)
    result_compiled = compiled_program(arg_0, arg_1, arg_2, arg_3, sentinel)
    print(' compile success')
except Exception as e:
    print(f' compile failed: {e}')

# Verify consistency
if 'result_eager' in locals() and 'result_compiled' in locals():
    if tf.reduce_all(tf.equal(result_eager, result_compiled)).numpy():
        print(' Eager and compiled results match.')
    else:
        print(' Divergence detected between eager and compiled results.')