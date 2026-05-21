import torch
import tensorflow as tf
import numpy as np

# Note: The provided "Similar API" tf.compat.v1.flags is a command-line flags parsing utility 
# and is not semantically similar to torch.unique (a tensor operation). 
# To preserve the core bug reproduction logic (tensor operations), this test case 
# uses the actual TensorFlow equivalent: tf.unique (specifically tf.compat.v1.unique 
# to match the namespace style requested).

# Set seeds for reproducibility
tf.random.set_seed(1012969)
np.random.seed(1012969)

# Define inputs
# torch.as_strided creates a view, here we create tensors with the same shape and data
arg_0 = tf.random.normal((2, 10), dtype=tf.float64)
arg_1 = tf.random.normal((10, 3), dtype=tf.float64)
sentinel = tf.constant(1.0, dtype=tf.float64)

def fuzzed_program(arg_0, arg_1, sentinel):
    var_node_3 = arg_0
    var_node_4 = arg_1
    var_node_2 = tf.matmul(var_node_3, var_node_4)

    # Replicate torch.arange(1, device=..., dtype=torch.int64)
    _inp_unique_wide = tf.range(1, dtype=tf.int64)

    # Replicate torch.unique using tf.compat.v1.unique
    # tf.compat.v1.unique returns (unique, idx). We need unique.
    # Note: torch.unique sorts, tf.unique does not. For input [0], it doesn't matter.
    _uniq_wide, _ = tf.compat.v1.unique(_inp_unique_wide)

    # Replicate .to(dtype)
    var_node_1 = tf.cast(_uniq_wide, var_node_2.dtype)

    # Replicate torch.full
    var_node_5 = tf.fill((1, 18), tf.constant(0.40330381448978797, dtype=tf.float64))

    # Replicate matmul
    var_node_0 = tf.matmul(var_node_1, var_node_5)

    # Replicate sentinel multiplication
    result = var_node_0 * sentinel

    # Replicate complex check
    if result.dtype.is_complex:
        result = tf.math.real(result)

    return result

# Run Eager
print("Running Eager...")
result_eager = fuzzed_program(arg_0, arg_1, sentinel)
print(f"Eager Result Shape: {result_eager.shape}")
print(f"Eager Result: {result_eager.numpy()}")

# Run Compiled (tf.function)
print("\nRunning Compiled...")
compiled_program = tf.function(fuzzed_program)
result_compiled = compiled_program(arg_0, arg_1, sentinel)
print(f"Compiled Result Shape: {result_compiled.shape}")
print(f"Compiled Result: {result_compiled.numpy()}")

# Verify
assert tf.reduce_all(tf.abs(result_eager - result_compiled) < 1e-6).numpy(), "Divergence detected!"
print(" Test Passed: Eager and Compiled results match.")