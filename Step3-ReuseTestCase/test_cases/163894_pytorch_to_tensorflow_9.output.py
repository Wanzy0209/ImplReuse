import torch
import tensorflow as tf

# Enable logging to observe any compilation behavior
tf.get_logger().setLevel('INFO')

def fuzzed_program(arg_0):
    """
    Adapted from the PyTorch fuzzed program to test tf.debugging.assert_type.
    The original program used torch.nonzero which caused a stride mismatch in 
    torch.compile. Here we replace that operation with tf.debugging.assert_type
    to verify type checking behavior in both eager and graph (compiled) modes.
    """
    # var_node_1 = arg_0 # size=(1, 2), dtype=int64
    var_node_1 = arg_0

    # var_node_5 = torch.full((1, 2), -66, dtype=torch.int32)
    var_node_5 = tf.fill((1, 2), -66)
    var_node_5 = tf.cast(var_node_5, tf.int32)

    # var_node_6 = torch.full((1, 2), 77, dtype=torch.int64)
    var_node_6 = tf.fill((1, 2), 77)
    var_node_6 = tf.cast(var_node_6, tf.int64)

    # var_node_4 = torch.ops.aten.add(var_node_5, var_node_6) # dtype=int32 in PyTorch
    # Note: TF promotes int32 + int64 to int64. We cast back to int32 to strictly follow 
    # the original trace's dtype, though standard TF behavior is promotion.
    var_node_4 = tf.add(var_node_5, tf.cast(var_node_6, tf.int32))

    # var_node_7 = torch.full((1, 2), -64, dtype=torch.int32)
    var_node_7 = tf.fill((1, 2), -64)
    var_node_7 = tf.cast(var_node_7, tf.int32)

    # var_node_3 = torch.ops.aten.mul(var_node_4, var_node_7)
    var_node_3 = tf.multiply(var_node_4, var_node_7)

    # var_node_9 = torch.full((3, 4), False, dtype=torch.bool)
    var_node_9 = tf.fill((3, 4), False)
    var_node_9 = tf.cast(var_node_9, tf.bool)

    # --- SIMILAR API USAGE ---
    # Original: var_node_8 = torch.nonzero(var_node_9)
    # Adapted: Use tf.debugging.assert_type to verify the type of var_node_9.
    # This replaces the data extraction with a type assertion.
    try:
        tf.debugging.assert_type(var_node_9, tf.bool, message="var_node_9 must be bool")
    except TypeError as e:
        print(f"Type assertion failed: {e}")
        return None

    # Since tf.debugging.assert_type returns a no_op (not a tensor), we cannot 
    # proceed with the original arithmetic: var_node_2 = var_node_3 + var_node_8.
    # We return the last valid tensor state to verify the graph executed correctly.
    return var_node_3

# Setup inputs
# arg_0: size=(1, 2), dtype=int64
arg_0 = tf.constant([[0, 1]], dtype=tf.int64)

print("--- Testing Eager Mode ---")
try:
    result_eager = fuzzed_program(arg_0)
    print(f" Eager execution successful. Result: {result_eager}")
except Exception as e:
    print(f" Eager execution failed: {e}")

print("\n--- Testing Compiled Mode (tf.function) ---")
try:
    # tf.function is the TensorFlow equivalent of torch.compile
    compiled_program = tf.function(fuzzed_program)
    result_compiled = compiled_program(arg_0)
    print(f" Compiled execution successful. Result: {result_compiled}")
except Exception as e:
    print(f" Compiled execution failed: {e}")

# Verify consistency
if result_eager is not None and result_compiled is not None:
    if tf.reduce_all(tf.equal(result_eager, result_compiled)):
        print("\n Consistency Check: Eager and Compiled results match.")
    else:
        print("\n Consistency Check: Eager and Compiled results diverged.")