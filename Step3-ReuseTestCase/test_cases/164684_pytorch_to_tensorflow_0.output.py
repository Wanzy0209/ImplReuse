import torch
import tensorflow as tf

# Set seed for reproducibility
tf.random.set_seed(19990)

def fuzzed_program(arg_0):
    # arg_0: size=(1,), dtype=int32
    # Note: tf.strings.as_string does not support bool inputs directly, 
    # so we adapt the input type from the original bool to int32 to ensure the test is runnable.
    var_node_1 = tf.squeeze(arg_0) # size=(), dtype=int32
    
    # Original PyTorch logic: var_node_0 = var_node_1.item() -> result = var_node_0 * sentinel
    # Adapted logic: Pass the scalar tensor to tf.strings.as_string.
    # In TensorFlow, extracting a Python scalar via .numpy() inside a tf.function is not allowed,
    # so we pass the 0-D tensor directly to the operation.
    result = tf.strings.as_string(var_node_1)
    return result

# Input generation
# Original: torch.randint(0, 2, (1,), dtype=torch.bool) > 0
# Adapted: Use int32 to ensure compatibility with as_string
arg_0 = tf.random.uniform((1,), minval=0, maxval=2, dtype=tf.int32)

# Eager execution
try:
    result_original = fuzzed_program(arg_0)
    print(' eager success')
except Exception as e:
    print(f' eager failed: {e}')

# Compiled execution
# Using tf.function to mimic torch.compile
compiled_program = tf.function(fuzzed_program, autograph=True)
try:
    result_compiled = compiled_program(arg_0)
    print(' compile success')
except Exception as e:
    print(f' compile failed: {e}')