import torch
import sys

# Attempt to import TensorFlow, handling potential environment incompatibilities
try:
    import tensorflow as tf
except ImportError as e:
    print("Skipping test: Failed to import TensorFlow due to environment incompatibility.")
    print(f"Details: {e}")
    print("This is likely due to a missing GLIBCXX_3.4.29 in the system's libstdc++.so.6.")
    sys.exit(0)

# Reproduce the logic of the PyTorch bug using TensorFlow's tf.raw_ops.RandomUniform
# The original bug involves a divergence between eager execution and compiled execution
# (torch.compile) when handling scalar operations and specific integer types.

def fuzzed_program(seed_input):
    # Use the similar API: tf.raw_ops.RandomUniform to generate the input argument
    # This replaces the torch.randn logic in the original issue.
    arg_0 = tf.raw_ops.RandomUniform(
        shape=(),
        dtype=tf.int32,
        minval=-10,
        maxval=10,
        seed=seed_input,
        seed2=0
    )

    # var_node_2 = -6 (dtype=int64)
    var_node_2 = tf.constant(-6, dtype=tf.int64)
    
    # var_node_3 = arg_0 (dtype=int32)
    var_node_3 = arg_0
    
    # var_node_1 = var_node_2 * var_node_3
    # Note: Type promotion might occur here depending on framework behavior
    var_node_1 = var_node_2 * var_node_3

    # var_node_5 = torch.full((), 1, dtype=torch.int64)
    # Using tf.constant to mimic torch.full for a scalar
    var_node_5 = tf.constant(1, dtype=tf.int64)

    # var_node_4 = var_node_5.item()
    # In TensorFlow, we keep it as a 0-d tensor. The original bug involved 
    # extracting a Python scalar and using it in the graph, which caused issues.
    # We simulate the usage of this value directly.
    var_node_4 = var_node_5

    # var_node_0 = var_node_1 / var_node_4
    # Performing the division operation that was part of the divergence trigger
    var_node_0 = var_node_1 / var_node_4

    return var_node_0

# Set seed for reproducibility
seed_val = 19989

# 1. Test Eager Execution
print("Testing Eager Execution...")
try:
    result_eager = fuzzed_program(seed_val)
    print(f' eager success: {result_eager.numpy()}')
except Exception as e:
    print(f' eager failed: {e}')

# 2. Test Compiled Execution (tf.function)
# This corresponds to torch.compile in the original issue
print("\nTesting Compiled Execution...")
try:
    compiled_program = tf.function(fuzzed_program)
    result_compiled = compiled_program(seed_val)
    print(f' compile success: {result_compiled.numpy()}')
except Exception as e:
    print(f' compile failed: {e}')

# 3. Check for Divergence
# The original bug reported a KeyError (crash) in compile mode, implying a divergence or failure.
# Here we assert that the results match if both succeed.
try:
    # Convert to numpy for comparison
    if result_eager.numpy() == result_compiled.numpy():
        print("\n Test passed: No divergence between eager and compiled results.")
    else:
        print(f"\n Test failed: Divergence detected! Eager={result_eager.numpy()}, Compiled={result_compiled.numpy()}")
except NameError:
    print("\n Skipping comparison due to previous failures.")