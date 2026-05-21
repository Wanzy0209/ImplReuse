import torch
import tensorflow as tf
import numpy as np

# Set random seed to match the original test case
tf.random.set_seed(974450504)

# API Under Test: tf.tpu.experimental.embedding.SGD
# We instantiate the optimizer configuration as requested.
# Note: This API is typically used within a TPU strategy, but we instantiate it here
# to verify its behavior and compatibility with the tensor operations.
optimizer_config = tf.tpu.experimental.embedding.SGD(
    learning_rate=0.01,
    momentum=0.0,
    nesterov=False
)

def fuzzed_program(arg_0, arg_1, sentinel):
    # var_node_3 = arg_0 # size=(17, 30, 17, 3), dtype=bool
    var_node_3 = arg_0

    # var_node_2 = torch.chunk(var_node_3, 3, dim=3)[0]
    # tf.split returns a list of tensors. We take the first one.
    # Original dim 3 has size 3, so splitting into 3 parts results in size 1 each.
    var_node_2 = tf.split(var_node_3, 3, axis=3)[0] # size=(17, 30, 17, 1)

    # var_node_5 = torch.full((17,), 3, dtype=torch.int64)
    var_node_5 = tf.fill([17], tf.cast(3, tf.int64)) # size=(17,), dtype=int64

    # var_node_6 = arg_1 # size=(15,), dtype=int64
    var_node_6 = arg_1

    # _input_size_var_node_4 = var_node_5.size(0)
    _input_size_var_node_4 = tf.shape(var_node_5)[0]

    # _index_var_node_4 = torch.randint(0, _input_size_var_node_4, (15,), device=var_node_5.device)
    # var_node_4 = torch.gather(var_node_5, 0, _index_var_node_4)
    # In TF, tf.gather(params, indices) gathers along axis 0 by default.
    _index_var_node_4 = tf.random.uniform((15,), minval=0, maxval=_input_size_var_node_4, dtype=tf.int32)
    # Cast indices to int64 to match PyTorch behavior if necessary, though gather handles int32
    var_node_4 = tf.gather(var_node_5, _index_var_node_4) # size=(15,)

    # _input_size_var_node_1 = var_node_2.size(0)
    _input_size_var_node_1 = tf.shape(var_node_2)[0]

    # _index_var_node_1 = torch.randint(0, _input_size_var_node_1, (15,), device=var_node_2.device)
    # var_node_1 = torch.index_select(var_node_2, 0, _index_var_node_1)
    _index_var_node_1 = tf.random.uniform((15,), minval=0, maxval=_input_size_var_node_1, dtype=tf.int32)
    var_node_1 = tf.gather(var_node_2, _index_var_node_1, axis=0) # size=(15, 30, 17, 1)

    # var_node_0 = torch.squeeze(var_node_1)
    # tf.squeeze removes dimensions of size 1.
    var_node_0 = tf.squeeze(var_node_1) # size=(15, 30, 17)

    # Ensure gradient computation by multiplying with sentinel
    result = var_node_0 * sentinel
    
    # Note: PyTorch checks is_complex(). TF tensors have a dtype.
    # We return the result directly.
    return result

# Sentinel tensor to ensure gradient computation
# In TF, we use a Variable to track gradients if needed, or just a tensor.
sentinel = tf.Variable(1.0, dtype=tf.float32)

# Setup inputs
# arg_0: size=(17, 30, 17, 3), dtype=bool
# PyTorch: torch.as_strided(torch.randint(0, 2, (26010,), dtype=torch.int8).bool(), ...)
# We create a random boolean tensor of the target shape.
arg_0 = tf.cast(tf.random.uniform((17, 30, 17, 3)) > 0.5, tf.bool)

# arg_1: size=(15,), dtype=int64
# PyTorch: torch.as_strided(torch.randint(5, 30, (15,)).to(torch.int64), ...)
arg_1 = tf.cast(tf.random.uniform((15,), minval=5, maxval=30), tf.int64)

args = (arg_0, arg_1, sentinel)

# 1. Eager Execution
print("Running Eager Execution...")
try:
    result_eager = fuzzed_program(*args)
    print(" eager success")
except Exception as e:
    print(f" eager failed: {e}")

# 2. Compiled Execution (tf.function)
# This mimics torch.compile
print("Running Compiled Execution...")
try:
    compiled_program = tf.function(fuzzed_program)
    result_compiled = compiled_program(*args)
    print(" compile success")
except Exception as e:
    print(f" compile failed: {e}")

# 3. Verify Divergence
# Check if eager and compiled results match
if 'result_eager' in locals() and 'result_compiled' in locals():
    # Use numpy for comparison as TF tensors might be on different devices or have different shapes
    if not np.array_equal(result_eager.numpy(), result_compiled.numpy()):
        print(" Divergence detected between eager and compiled results!")
    else:
        print(" Results match between eager and compiled execution")