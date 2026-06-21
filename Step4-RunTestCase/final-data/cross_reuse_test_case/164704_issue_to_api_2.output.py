import sys

# Attempt to import TensorFlow, handling environment incompatibilities
try:
    import tensorflow as tf
except ImportError as e:
    # Check for the specific GLIBCXX error mentioned in the traceback
    if "GLIBCXX" in str(e):
        print("SKIP: Cannot run test due to environment incompatibility (GLIBCXX version too old for TensorFlow).")
        print(f"Error details: {e}")
        sys.exit(0)
    else:
        # Re-raise if it's a different import error
        raise

def fuzzed_program(arg_0, arg_1):
    # Replicating the logic from the PyTorch bug report to test execution modes
    # var_node_4 = torch.full((2, 3), 3, dtype=torch.int16)
    var_node_4 = tf.fill((2, 3), tf.cast(3, tf.int16))
    
    # var_node_3 = torch.unique(var_node_4)
    # Note: tf.unique flattens the input.
    var_node_3 = tf.unique(var_node_4)[0]
    
    # var_node_2 = torch.squeeze(var_node_3)
    var_node_2 = tf.squeeze(var_node_3)
    
    # var_node_7 = arg_0, var_node_8 = arg_1
    var_node_7 = tf.cast(arg_0, tf.int16)
    var_node_8 = tf.cast(arg_1, tf.int16)
    
    # var_node_6 = torch.sub(var_node_7, var_node_8)
    var_node_6 = tf.subtract(var_node_7, var_node_8)
    
    # var_node_10 = torch.full((1,), 3, dtype=torch.int16)
    var_node_10 = tf.fill((1,), tf.cast(3, tf.int16))
    
    # var_node_9 = torch.squeeze(var_node_10)
    var_node_9 = tf.squeeze(var_node_10)
    
    # var_node_5 = torch.add(var_node_6, var_node_9)
    var_node_5 = tf.add(var_node_6, var_node_9)
    
    # var_node_1 = torch.div(var_node_2, var_node_5)
    # The original bug involved a type mismatch here during compilation.
    var_node_1 = tf.div(var_node_2, var_node_5)
    
    # Leveraging the similar API: tf.compat.v1.executing_eagerly
    # This checks the execution mode, which is central to the "Eager/Compile Divergence" bug.
    is_eager = tf.compat.v1.executing_eagerly()
    
    return var_node_1, is_eager

# Setup inputs similar to the original bug
arg_0 = tf.constant(10, dtype=tf.int16)
arg_1 = tf.constant(2, dtype=tf.int16)

# 1. Eager Execution
result_val, is_eager = fuzzed_program(arg_0, arg_1)
assert is_eager == True, "Expected eager execution to be True"
print(f' eager success: {result_val.numpy()}, is_eager={is_eager}')

# 2. Compiled/Graph Execution (Analogous to torch.compile)
compiled_program = tf.function(fuzzed_program)
result_val_compiled, is_eager_compiled = compiled_program(arg_0, arg_1)
assert is_eager_compiled == False, "Expected eager execution to be False in graph mode"
print(f' compile success: {result_val_compiled.numpy()}, is_eager={is_eager_compiled}')