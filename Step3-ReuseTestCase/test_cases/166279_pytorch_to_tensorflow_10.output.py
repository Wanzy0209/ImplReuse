import torch
import tensorflow as tf
import numpy as np

# Enable numpy behavior for tf.experimental.numpy
tf.experimental.numpy.experimental_enable_numpy_behavior()

# Set seed for reproducibility, mimicking torch.manual_seed(1166094474)
np.random.seed(1166094474)
tf.random.set_seed(1166094474)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, sentinel):
    # Adaptation logic:
    # The original PyTorch test uses torch.chunk to split tensors.
    # We adapt this to use tf.experimental.numpy.tril, which operates on 2D+ tensors.
    # We focus on arg_2 which is 2D (shape 6, 4).
    
    # Original: var_node_10 = arg_2
    # Original: var_node_9 = torch.chunk(var_node_10, 4, dim=1)[0]
    # Adapted: Apply tril to the 2D tensor arg_2
    var_node_tril = tf.experimental.numpy.tril(arg_2)
    
    # Ensure gradient computation by multiplying with sentinel
    result = var_node_tril * sentinel
    return result

# Sentinel tensor to ensure gradient computation
sentinel = tf.constant(1.0)

# Create inputs mimicking the PyTorch test case structure
# PyTorch arg_0: size=(12,), stride=(1,), dtype=bool
arg_0 = tf.constant(np.random.randint(0, 2, (12,), dtype=np.bool_))

# PyTorch arg_1: size=(10,), stride=(1,), dtype=int64
arg_1 = tf.constant(np.random.randint(5, 30, (10,), dtype=np.int64))

# PyTorch arg_2: size=(6, 4), stride=(4, 1), dtype=bool
# We create a (6, 4) bool tensor. 
# Note: TF tensors are immutable, so we don't set strides explicitly, 
# but we match the shape and dtype.
arg_2 = tf.constant(np.random.randint(0, 2, (6, 4), dtype=np.bool_))

# PyTorch arg_3: size=(2,), stride=(1,), dtype=bool
arg_3 = tf.constant(np.random.randint(0, 2, (2,), dtype=np.bool_))

args = (arg_0, arg_1, arg_2, arg_3, sentinel)

# Run Eager
print("Running eager...")
try:
    result_original = fuzzed_program(*args)
    print(' eager success')
except Exception as e:
    print(f' eager failed: {e}')
    result_original = None

# Run Compiled (tf.function mimics torch.compile)
print("Running compiled...")
compiled_program = tf.function(fuzzed_program)
try:
    result_compiled = compiled_program(*args)
    print(' compile success')
except Exception as e:
    print(f' compile failed: {e}')
    result_compiled = None

# Verify results
if result_original is not None and result_compiled is not None:
    if tf.reduce_all(tf.equal(result_original, result_compiled)).numpy():
        print(' results match')
    else:
        print(' results diverge')