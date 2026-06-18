import torch
import tensorflow as tf
import numpy as np

# Set seeds for reproducibility
np.random.seed(0)
tf.random.set_seed(0)

def foo(input):
    # Define the RNN cells for the bidirectional dynamic RNN
    # Using BasicRNNCell for simplicity, similar to how interpolate is a basic functional op
    cell_fw = tf.compat.v1.nn.rnn_cell.BasicRNNCell(num_units=10)
    cell_bw = tf.compat.v1.nn.rnn_cell.BasicRNNCell(num_units=10)
    
    # Call the target API: tf.compat.v1.nn.bidirectional_dynamic_rnn
    # This replaces torch.nn.functional.interpolate
    outputs, _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(
        cell_fw,
        cell_bw,
        input,
        dtype=tf.float64
    )
    
    # Post-processing to mimic the original test case structure
    # outputs is a tuple (output_fw, output_bw), each shape (batch, time, units)
    # Concatenate them to get a single tensor
    combined = tf.concat(outputs, axis=2) # Shape (1, 40, 20)
    
    # Mimic the squeeze and argmin operations from the PyTorch code
    squeeze = tf.squeeze(combined, 0) # Shape (40, 20)
    argmin = tf.argmin(squeeze, axis=1) # Shape (40)
    return argmin

# Prepare input data
# PyTorch input was (1, 40, 1, 1). Mapping to RNN input (batch, time, features) -> (1, 40, 1)
x = np.random.uniform(0, 10, size=(1, 40, 1)).astype(np.float64)

# Eager execution (equivalent to PyTorch eager mode)
eager_res = foo(tf.constant(x))

# Compiled execution (equivalent to torch.compile)
# In TensorFlow, tf.function compiles the graph
compiled_foo = tf.function(foo)
compile_res = compiled_foo(tf.constant(x))

# Verify consistency between eager and compiled results
# This mirrors the torch.testing.assert_close in the original bug report
try:
    np.testing.assert_array_equal(eager_res.numpy(), compile_res.numpy())
    print("Test passed: Eager and Compiled results are consistent.")
except AssertionError as e:
    print("Test failed: Inconsistency detected between eager and compiled modes.")
    raise e