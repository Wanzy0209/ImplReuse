import torch
import tensorflow as tf
import numpy as np

# Set seed for reproducibility
tf.random.set_seed(1012969)

# Sentinel variable to mimic the gradient check logic
sentinel = tf.Variable(1.0, dtype=tf.float64)

def fuzzed_program(sp_arg_0, sp_arg_1, dense_arg, sentinel_var):
    # Original PyTorch logic:
    # 1. torch.unique (manipulates tensor)
    # 2. torch.matmul (uses result)
    #
    # Adapted TensorFlow logic for tf.sparse.concat:
    # 1. tf.sparse.concat (manipulates sparse tensors)
    # 2. tf.sparse.to_dense + tf.matmul (uses result)

    # Perform sparse concatenation along axis 0
    # Inputs are (1, 10), output will be (2, 10)
    sp_concat = tf.sparse.concat(sp_inputs=[sp_arg_0, sp_arg_1], axis=0)

    # Convert to dense to perform matmul
    dense_concat = tf.sparse.to_dense(sp_concat)

    # Perform matmul
    # (2, 10) @ (10, 3) -> (2, 3)
    result = tf.matmul(dense_concat, dense_arg)

    # Multiply by sentinel
    result = result * sentinel_var
    return result

# Create input sparse tensors
# Shape (1, 10)
indices_0 = [[0, i] for i in range(10)]
values_0 = tf.cast(tf.range(10), tf.float64)
shape_0 = [1, 10]
sp_input_0 = tf.sparse.SparseTensor(indices_0, values_0, shape_0)

indices_1 = [[0, i] for i in range(10)]
values_1 = tf.cast(tf.range(10, 20), tf.float64)
shape_1 = [1, 10]
sp_input_1 = tf.sparse.SparseTensor(indices_1, values_1, shape_1)

# Create dense argument for matmul
# Shape (10, 3)
dense_arg = tf.random.uniform((10, 3), dtype=tf.float64)

# Test Eager Execution
print("Testing Eager Execution...")
try:
    result_eager = fuzzed_program(sp_input_0, sp_input_1, dense_arg, sentinel)
    print(" Eager success. Shape:", result_eager.shape)
except Exception as e:
    print(" Eager failed:", e)

# Test Compiled Execution (tf.function)
print("\nTesting Compiled Execution...")
compiled_program = tf.function(fuzzed_program)
try:
    result_compiled = compiled_program(sp_input_0, sp_input_1, dense_arg, sentinel)
    print(" Compile success. Shape:", result_compiled.shape)
except Exception as e:
    print(" Compile failed:", e)

# Verify consistency
if 'result_eager' in locals() and 'result_compiled' in locals():
    if tf.reduce_all(tf.equal(result_eager, result_compiled)).numpy():
        print(" Results match.")
    else:
        print(" Results diverge.")