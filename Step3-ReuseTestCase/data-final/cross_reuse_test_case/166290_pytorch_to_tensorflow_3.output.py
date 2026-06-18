import torch
import tensorflow as tf
import numpy as np

# Setup
tf.random.set_seed(974450504)

# Define inputs
# arg_0: size=(17, 30, 17, 3), dtype=bool
arg_0 = tf.cast(tf.random.uniform((17, 30, 17, 3), 0, 2, dtype=tf.int32), tf.bool)

# arg_1: size=(15,), dtype=int64
arg_1 = tf.random.uniform((15,), 5, 30, dtype=tf.int64)

# Sentinel tensor to ensure gradient computation
sentinel = tf.Variable(1.0, dtype=tf.float32)

def fuzzed_program(arg_0, arg_1, sentinel):
    # var_node_2 = torch.chunk(var_node_3, 3, dim=3)[0]
    var_node_2 = tf.split(arg_0, 3, axis=3)[0] # size=(17, 30, 17, 1)

    # var_node_5 = torch.full((17,), 3, dtype=torch.int64)
    var_node_5 = tf.fill((17,), 3)

    # var_node_4 = torch.gather(var_node_5, 0, _index_var_node_4)
    _input_size_var_node_4 = tf.shape(var_node_5)[0]
    _index_var_node_4 = tf.random.uniform((15,), 0, _input_size_var_node_4, dtype=tf.int32)
    var_node_4 = tf.gather(var_node_5, _index_var_node_4)

    # var_node_1 = torch.index_select(var_node_2, 0, _index_var_node_1)
    _input_size_var_node_1 = tf.shape(var_node_2)[0]
    _index_var_node_1 = tf.random.uniform((15,), 0, _input_size_var_node_1, dtype=tf.int32)
    var_node_1 = tf.gather(var_node_2, _index_var_node_1, axis=0) # size=(15, 30, 17, 1)

    # Original API: torch.squeeze(var_node_1)
    # Adapted API: tf.experimental.numpy.tril(var_node_1)
    var_node_0 = tf.experimental.numpy.tril(var_node_1)

    # Ensure gradient computation by multiplying with sentinel
    result = var_node_0 * sentinel

    # Handle complex numbers if necessary
    if result.dtype.is_complex:
        result = tf.math.real(result)

    return result

# Test Eager
print("Testing Eager Mode...")
try:
    result_eager = fuzzed_program(arg_0, arg_1, sentinel)
    print(" eager success")
except Exception as e:
    print(f" eager failed: {e}")

# Test Compiled
print("Testing Compiled Mode (tf.function)...")
try:
    compiled_program = tf.function(fuzzed_program)
    result_compiled = compiled_program(arg_0, arg_1, sentinel)
    print(" compile success")
except Exception as e:
    print(f" compile failed: {e}")