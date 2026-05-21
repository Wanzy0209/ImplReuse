import torch
import tensorflow as tf

# Configure TensorFlow to mimic the behavior of the original test
# The original bug was related to dynamic shape handling in compilation.
# We use tf.function to test the compiled (graph) mode.

def fuzzed_program(arg_0, arg_1, sentinel):
    # var_node_3 = arg_0 # size=(2, 10), dtype=float64
    # var_node_4 = arg_1 # size=(10, 3), dtype=float64
    # var_node_2 = torch.matmul(var_node_3, var_node_4)
    var_node_2 = tf.linalg.matmul(arg_0, arg_1) # size=(2, 3), dtype=float64

    # _inp_unique_wide = torch.arange(1, device=var_node_2.device, dtype=torch.int64)
    _inp_unique_wide = tf.range(1, dtype=tf.int64)

    # Original: _uniq_wide = torch.unique(_inp_unique_wide)
    # Adaptation: Use tf.math.add as the similar API identified.
    # We add 0 to the tensor to mimic a transformation that preserves shape (1,)
    # but changes the operation from unique to add.
    _uniq_wide = tf.math.add(_inp_unique_wide, 0)

    # var_node_1 = _uniq_wide.to(var_node_2.dtype)
    var_node_1 = tf.cast(_uniq_wide, tf.float64) # size=(1,), dtype=float64

    # var_node_5 = torch.full((1, 18), 0.40330381448978797, dtype=torch.float64)
    var_node_5 = tf.fill((1, 18), tf.constant(0.40330381448978797, dtype=tf.float64)) # size=(1, 18), dtype=float64

    # var_node_0 = torch.matmul(var_node_1, var_node_5)
    # Note: PyTorch matmul with (1,) and (1, 18) results in (18,).
    # TensorFlow matmul with (1,) and (1, 18) results in (1, 18).
    # We use tf.linalg.matmul here.
    var_node_0 = tf.linalg.matmul(var_node_1, var_node_5)

    # Ensure gradient computation by multiplying with sentinel
    result = var_node_0 * sentinel
    
    # Check for complex type
    if result.dtype == tf.complex64 or result.dtype == tf.complex128:
        result = tf.math.real(result)
        
    return result

# Sentinel tensor
sentinel = tf.constant(1.0)

# Setup inputs
# Using tf.random.normal to approximate torch.randn
# Original used as_strided, but random generation is sufficient for logic reproduction
tf.random.set_seed(1012969)
arg_0 = tf.random.normal((2, 10), dtype=tf.float64)
arg_1 = tf.random.normal((10, 3), dtype=tf.float64)

args = (arg_0, arg_1, sentinel)

# Test Eager Execution
print('Running eager execution...')
try:
    result_original = fuzzed_program(*args)
    print(f' eager success, shape: {result_original.shape}')
except Exception as e:
    print(f' eager failed: {e}')

# Test Compiled Execution (tf.function)
print('Running compiled execution...')
try:
    compiled_program = tf.function(fuzzed_program)
    result_compiled = compiled_program(*args)
    print(f' compile success, shape: {result_compiled.shape}')
except Exception as e:
    print(f' compile failed: {e}')

# Verify consistency
if 'result_original' in locals() and 'result_compiled' in locals():
    if tf.reduce_all(tf.abs(result_original - result_compiled) < 1e-6):
        print(' Results match between eager and compiled.')
    else:
        print(' Results diverge between eager and compiled.')