import torch
import tensorflow as tf

# Set seed for reproducibility
tf.random.set_seed(1215252001)

def fuzzed_program(arg_0, arg_1, arg_2, sentinel):
    # Use tf.name_scope to group operations, corresponding to the API under test
    with tf.name_scope("fuzzed_logic"):
        # var_node_4 = arg_0
        # var_node_3 = torch.squeeze(var_node_4)
        # Input arg_0 shape: (9, 1, 15, 4) -> Output: (9, 15, 4)
        var_node_3 = tf.squeeze(arg_0)

        # var_node_2 = torch.chunk(var_node_3, 4, dim=2)[0]
        # Split dim 2 (size 4) into 4 chunks of size 1. Take the first one.
        # Shape: (9, 15, 1)
        var_node_2 = tf.split(var_node_3, 4, axis=2)[0]

        # var_node_1 = torch.squeeze(var_node_2)
        # Shape: (9, 15)
        var_node_1 = tf.squeeze(var_node_2)

        # var_node_7 = arg_1
        # _input_size_var_node_6 = var_node_7.size(0)
        _input_size_var_node_6 = tf.shape(arg_1)[0]

        # _index_var_node_6 = torch.randint(0, _input_size_var_node_6, (18, 15), ...)
        # Generate random indices for gather
        _index_var_node_6 = tf.random.uniform(
            shape=(18, 15),
            minval=0,
            maxval=_input_size_var_node_6,
            dtype=tf.int64
        )

        # var_node_6 = torch.gather(var_node_7, 0, _index_var_node_6)
        # Gather along axis 0
        var_node_6 = tf.gather(arg_1, _index_var_node_6, axis=0)

        # var_node_5 = torch.chunk(var_node_6, 2, dim=0)[0]
        # Split dim 0 (size 18) into 2 chunks of size 9. Take the first one.
        # Shape: (9, 15)
        var_node_5 = tf.split(var_node_6, 2, axis=0)[0]

        # var_node_0 = torch.mul(var_node_1, var_node_5)
        var_node_0 = tf.multiply(var_node_1, var_node_5)

        # result = var_node_0 * sentinel
        result = var_node_0 * sentinel
        
        return result

# --- Input Construction ---

# arg_0: size=(9, 1, 15, 4), stride=(60, 60, 0, 1), dtype=int32
# The stride (..., 0, ...) on dim 2 implies broadcasting.
# We mimic this by creating a smaller tensor and broadcasting it.
base_tensor_0 = tf.random.uniform((9, 1, 1, 4), minval=5, maxval=30, dtype=tf.int32)
arg_0 = tf.broadcast_to(base_tensor_0, (9, 1, 15, 4))

# arg_1: size=(20, 15), stride=(15, 1), dtype=int64
arg_1 = tf.random.uniform((20, 15), minval=5, maxval=30, dtype=tf.int64)

# arg_2: size=(18, 15), stride=(15, 1), dtype=int64
# Note: arg_2 is unused in the function body but part of the signature.
arg_2 = tf.random.uniform((18, 15), minval=5, maxval=30, dtype=tf.int64)

# Sentinel tensor
sentinel = tf.constant(1.0)

args = (arg_0, arg_1, arg_2, sentinel)

# --- Execution ---

# 1. Run in Eager Mode
print("Running Eager Mode...")
result_eager = fuzzed_program(*args)
print(f" eager success, result shape: {result_eager.shape}")

# 2. Run in Compiled Mode (tf.function)
# This corresponds to torch.compile in the original issue.
print("Running Compiled Mode...")
compiled_program = tf.function(fuzzed_program)
result_compiled = compiled_program(*args)
print(f" compile success, result shape: {result_compiled.shape}")

# --- Verification ---
# The original bug was a crash (LoweringException). 
# Reaching this point implies the API handled the operations correctly.
# We check shape consistency to ensure logic preservation.
assert result_eager.shape == result_compiled.shape
print(" Verification Passed: Eager and Compiled modes produced consistent shapes.")