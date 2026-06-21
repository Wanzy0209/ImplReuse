import torch
import tensorflow as tf

# Set seed for reproducibility
tf.random.set_seed(974450504)

def fuzzed_program(arg_0, arg_1, sentinel):
    # var_node_3 = arg_0 # size=(17, 30, 17, 3), dtype=bool
    var_node_3 = arg_0

    # var_node_2 = torch.chunk(var_node_3, 3, dim=3)[0]
    # Splits the last dimension (size 3) into 3 chunks of size 1, takes the first.
    # Result shape: (17, 30, 17, 1)
    var_node_2 = tf.split(var_node_3, 3, axis=-1)[0]

    # var_node_5 = torch.full((17,), 3, dtype=torch.int64)
    var_node_5 = tf.fill((17,), 3)

    # _input_size_var_node_4 = var_node_5.size(0)
    _input_size_var_node_4 = tf.shape(var_node_5)[0]

    # _index_var_node_4 = torch.randint(0, _input_size_var_node_4, (15,), device=var_node_5.device)
    _index_var_node_4 = tf.random.uniform((15,), minval=0, maxval=_input_size_var_node_4, dtype=tf.int32)

    # var_node_4 = torch.gather(var_node_5, 0, _index_var_node_4)
    var_node_4 = tf.gather(var_node_5, _index_var_node_4)

    # _input_size_var_node_1 = var_node_2.size(0)
    _input_size_var_node_1 = tf.shape(var_node_2)[0]

    # _index_var_node_1 = torch.randint(0, _input_size_var_node_1, (15,), device=var_node_2.device)
    _index_var_node_1 = tf.random.uniform((15,), minval=0, maxval=_input_size_var_node_1, dtype=tf.int32)

    # var_node_1 = torch.index_select(var_node_2, 0, _index_var_node_1)
    # Result shape: (15, 30, 17, 1)
    var_node_1 = tf.gather(var_node_2, _index_var_node_1, axis=0)

    # --- TARGET API CHANGE ---
    # Original: var_node_0 = torch.squeeze(var_node_1)
    # Target: var_node_0 = tf.keras.ops.tril(var_node_1)
    # Note: tril operates on the last two dimensions.
    var_node_0 = tf.keras.ops.tril(var_node_1)

    # Ensure gradient computation by multiplying with sentinel
    result = var_node_0 * sentinel
    return result

# Sentinel tensor to ensure gradient computation
sentinel = tf.constant(1.0)

# arg_0: size=(17, 30, 17, 3), dtype=bool
# Using random boolean tensor to mimic the input
arg_0 = tf.random.uniform((17, 30, 17, 3), minval=0, maxval=2, dtype=tf.int32) > 0
arg_0 = tf.cast(arg_0, tf.bool)

# arg_1: size=(15,), dtype=int64
arg_1 = tf.random.uniform((15,), minval=5, maxval=30, dtype=tf.int64)

args = (arg_0, arg_1, sentinel)

# Test Eager
try:
    result_original = fuzzed_program(*args)
    print(' eager success')
except Exception as e:
    print(f' eager failed: {e}')

# Test Compiled (tf.function)
try:
    compiled_program = tf.function(fuzzed_program)
    result_compiled = compiled_program(*args)
    print(' compile success')
except Exception as e:
    print(f' compile failed: {e}')