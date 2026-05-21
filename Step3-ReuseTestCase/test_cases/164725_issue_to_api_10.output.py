import torch
import tensorflow as tf
import numpy as np

def fuzzed_program(sentinel):
    # Replicate the tensor manipulation logic from the PyTorch issue
    # var_node_4 = torch.full((6,), True, dtype=torch.bool)
    var_node_4 = tf.cast(tf.fill([6], 1), dtype=tf.bool)
    
    # var_node_3 = torch.reshape(var_node_4, [2, 3])
    var_node_3 = tf.reshape(var_node_4, [2, 3])
    
    # _x_ms = torch.arange(max(1, 1), device=var_node_3.device).to(var_node_3.dtype)
    # max(1, 1) is 1. arange(1) is [0]. to(bool) is [False].
    _x_ms = tf.cast(tf.range(1), dtype=tf.bool)
    
    # _mask_ms = torch.zeros_like(_x_ms, dtype=torch.bool)
    # _mask_ms[:1] = True
    # _x_ms is shape (1,), so _mask_ms becomes [True]
    _mask_ms = tf.tensor_scatter_nd_update(tf.zeros_like(_x_ms, dtype=tf.bool), [[0]], [True])
    
    # var_node_2 = torch.masked_select(_x_ms, _mask_ms)
    var_node_2 = tf.boolean_mask(_x_ms, _mask_ms)
    
    # var_node_1 = torch.squeeze(var_node_2)
    var_node_1 = tf.squeeze(var_node_2)
    
    # Original API: var_node_0 = var_node_1.item()
    # Similar API: tf.keras.ops.exp
    # We cast to float32 as exp expects float inputs, similar to how item() extracts a value for computation.
    var_node_0 = tf.keras.ops.exp(tf.cast(var_node_1, tf.float32))
    
    # Ensure gradient computation by multiplying with sentinel
    result = var_node_0 * sentinel
    return result

# Sentinel variable to ensure gradient computation
sentinel = tf.Variable(1.0)

# Test Eager Execution
print("Testing Eager Execution...")
with tf.GradientTape() as tape:
    result_eager = fuzzed_program(sentinel)
grad_eager = tape.gradient(result_eager, sentinel)
print(f" Eager Result: {result_eager.numpy()}, Gradient: {grad_eager.numpy()}")

# Test Compiled Execution (tf.function is analogous to torch.compile)
print("Testing Compiled Execution...")
compiled_program = tf.function(fuzzed_program)
with tf.GradientTape() as tape:
    result_compiled = compiled_program(sentinel)
grad_compiled = tape.gradient(result_compiled, sentinel)
print(f" Compiled Result: {result_compiled.numpy()}, Gradient: {grad_compiled.numpy()}")

# Assertions to check for divergence (DDE)
assert np.allclose(result_eager.numpy(), result_compiled.numpy()), "Output divergence detected between eager and compiled modes!"
assert np.allclose(grad_eager.numpy(), grad_compiled.numpy()), "Gradient divergence detected between eager and compiled modes!"

print(" Test Passed: No divergence detected.")