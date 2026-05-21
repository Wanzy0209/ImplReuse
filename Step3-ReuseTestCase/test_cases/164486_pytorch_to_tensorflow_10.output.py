import torch
import tensorflow as tf
import numpy as np

# Set seed for reproducibility
tf.random.set_seed(238)

# Define inputs
# PyTorch: arg_0 = torch.as_strided(torch.randn(1).to(torch.int16), (), ())
# TF equivalent: 0-dim int16 tensor
arg_0 = tf.random.uniform(shape=[], minval=-10, maxval=10, dtype=tf.int16)

# Sentinel tensor to ensure gradient computation
# PyTorch: sentinel = torch.tensor(1.0, requires_grad=True)
sentinel = tf.Variable(1.0, dtype=tf.float32)

def fuzzed_program(arg_0, sentinel):
    # var_node_2 = torch.full((), 1, dtype=torch.int16)
    var_node_2 = tf.constant(1, dtype=tf.int16)

    # var_node_3 = arg_0
    var_node_3 = arg_0

    # var_node_1 = torch.add(var_node_2, var_node_3)
    var_node_1 = tf.add(var_node_2, var_node_3)

    # var_node_5 = torch.full((1,), 3, dtype=torch.int16)
    var_node_5 = tf.constant([3], dtype=tf.int16)

    # var_node_4 = torch.squeeze(var_node_5)
    var_node_4 = tf.squeeze(var_node_5)

    # var_node_0 = torch.div(var_node_1, var_node_4)
    # ADAPTATION: Replace torch.div with the target API tf.experimental.numpy.add
    var_node_0 = tf.experimental.numpy.add(var_node_1, var_node_4)

    # Ensure gradient computation by multiplying with sentinel
    # Cast to float32 to allow multiplication with float sentinel
    result = tf.cast(var_node_0, tf.float32) * sentinel
    return result

# 1. Eager Execution
with tf.GradientTape() as tape:
    out_eager = fuzzed_program(arg_0, sentinel)
grads_eager = tape.gradient(out_eager, sentinel)
print('Eager Success! ')

# 2. Compiled Execution
# PyTorch: torch.compile(..., fullgraph=True, dynamic=True)
# TF: tf.function with jit_compile=True
compiled_program = tf.function(fuzzed_program, jit_compile=True)

with tf.GradientTape() as tape:
    out_compiled = compiled_program(arg_0, sentinel)
grads_compiled = tape.gradient(out_compiled, sentinel)
print('Compile Success! ')

# Comparison logic
out_eager_sum = tf.reduce_sum(out_eager)
out_compiled_sum = tf.reduce_sum(out_compiled)

diff = tf.abs(out_eager_sum - out_compiled_sum)
rel_diff = diff / (tf.abs(out_eager_sum) + 1e-12) * 100

print(f'Relative diff (sum): {rel_diff.numpy():.6f}%')

if rel_diff > 5 and diff > 1:
    print(f' Forward output sums differ significantly (relative and absolute)!')
    print('out_eager_sum:', out_eager_sum.numpy())
    print('out_compiled_sum:', out_compiled_sum.numpy())
    print('Absolute diff:', diff.numpy())
    print('Relative diff (%):', rel_diff.numpy())
    import sys
    sys.exit(1)