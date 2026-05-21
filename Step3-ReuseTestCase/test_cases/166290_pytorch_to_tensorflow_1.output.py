import torch
import tensorflow as tf
import tensorflow.experimental.numpy as tnp

# Set seed for reproducibility
tf.random.set_seed(974450504)

def fuzzed_program(arg_0, arg_1, sentinel):
    # var_node_3 = arg_0 # size=(17, 30, 17, 3), dtype=bool
    # var_node_2 = torch.chunk(var_node_3, 3, dim=3)[0] # size=(17, 30, 17, 1)
    # TensorFlow equivalent: split along axis 3
    var_node_2 = tf.split(arg_0, 3, axis=3)[0]

    # var_node_5 = torch.full((17,), 3, dtype=torch.int64) # size=(17,), dtype=int64
    var_node_5 = tf.fill((17,), 3)
    var_node_5 = tf.cast(var_node_5, tf.int64)

    # var_node_6 = arg_1 # size=(15,), dtype=int64

    # _input_size_var_node_4 = var_node_5.size(0)
    _input_size_var_node_4 = tf.shape(var_node_5)[0]

    # _index_var_node_4 = torch.randint(0, _input_size_var_node_4, (15,), device=var_node_5.device)
    _index_var_node_4 = tf.random.uniform((15,), minval=0, maxval=_input_size_var_node_4, dtype=tf.int64)

    # var_node_4 = torch.gather(var_node_5, 0, _index_var_node_4) # size=(15,), dtype=int64
    var_node_4 = tf.gather(var_node_5, _index_var_node_4, axis=0)

    # _input_size_var_node_1 = var_node_2.size(0)
    _input_size_var_node_1 = tf.shape(var_node_2)[0]

    # _index_var_node_1 = torch.randint(0, _input_size_var_node_1, (15,), device=var_node_2.device)
    _index_var_node_1 = tf.random.uniform((15,), minval=0, maxval=_input_size_var_node_1, dtype=tf.int64)

    # var_node_1 = torch.index_select(var_node_2, 0, _index_var_node_1) # size=(15, 30, 17, 1)
    var_node_1 = tf.gather(var_node_2, _index_var_node_1, axis=0)

    # var_node_0 = torch.squeeze(var_node_1) 
    # ADAPTED: Use tf.experimental.numpy.triu instead of squeeze
    # triu preserves shape, so output remains (15, 30, 17, 1)
    var_node_0 = tnp.triu(var_node_1)

    # Ensure gradient computation by multiplying with sentinel
    result = var_node_0 * sentinel
    return result

# Sentinel tensor to ensure gradient computation
sentinel = tf.constant(1.0)

# Setup inputs
# arg_0: size=(17, 30, 17, 3), dtype=bool
arg_0 = tf.random.uniform((17, 30, 17, 3), minval=0, maxval=2, dtype=tf.int32) > 0
# arg_1: size=(15,), dtype=int64
arg_1 = tf.random.uniform((15,), minval=5, maxval=30, dtype=tf.int64)

args = (arg_0, arg_1, sentinel)

# Run Eager
print('Running eager mode...')
result_original = fuzzed_program(*args)
print(f' eager success, shape: {result_original.shape}, dtype: {result_original.dtype}')

# Run Compiled (Graph Mode)
print('Running compiled mode (tf.function)...')
compiled_program = tf.function(fuzzed_program)
result_compiled = compiled_program(*args)
print(f' compile success, shape: {result_compiled.shape}, dtype: {result_compiled.dtype}')

# Assertions
# triu preserves the input shape (15, 30, 17, 1), unlike squeeze which would reduce it to (15, 30, 17)
assert result_original.shape == (15, 30, 17, 1), f"Expected shape (15, 30, 17, 1), got {result_original.shape}"
assert result_compiled.shape == (15, 30, 17, 1), f"Expected shape (15, 30, 17, 1), got {result_compiled.shape}"
assert result_original.dtype == tf.float32, f"Expected float32, got {result_original.dtype}"