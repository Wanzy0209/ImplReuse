import torch
import tensorflow as tf
import numpy as np

def test_tf_math_add_compile_divergence():
    """
    Test case adapted from PyTorch Issue 164685.
    Original Issue: Eager/Compile divergence with scalar operations and mixed dtypes.
    Similar API: tf.compat.v1.math.add
    
    This test checks if tf.compat.v1.math.add behaves consistently between
    eager execution and tf.function (compiled) execution when handling
    mixed scalar types (int32, int64) and subsequent division operations.
    """
    
    # Setup inputs mirroring the original bug report
    # arg_0 is int32
    arg_0 = tf.constant(5, dtype=tf.int32)
    # Sentinel for gradient tracking
    sentinel = tf.Variable(1.0, dtype=tf.float32)

    def fuzzed_program(arg_0, sentinel):
        # var_node_2 = -6 (dtype=int64)
        var_node_2 = tf.constant(-6, dtype=tf.int64)
        
        # var_node_3 = arg_0 (dtype=int32)
        var_node_3 = arg_0
        
        # Original logic: var_node_1 = var_node_2 * var_node_3
        # Adapted logic using similar API: tf.compat.v1.math.add
        # We cast var_node_2 to int32 to match the implicit promotion behavior 
        # or to ensure type compatibility for the operation.
        var_node_2_casted = tf.cast(var_node_2, tf.int32)
        var_node_1 = tf.compat.v1.math.add(var_node_2_casted, var_node_3)
        
        # var_node_5 = torch.full((), 1, dtype=torch.int64)
        var_node_5 = tf.constant(1, dtype=tf.int64)
        
        # var_node_0 = var_node_1 / var_node_4
        # Performing division. Casting to float to match standard division behavior.
        var_node_0 = tf.cast(var_node_1, tf.float64) / tf.cast(var_node_5, tf.float64)
        
        # Ensure gradient computation by multiplying with sentinel
        result = var_node_0 * sentinel
        return result

    # 1. Eager Execution
    with tf.GradientTape() as tape_eager:
        result_eager = fuzzed_program(arg_0, sentinel)
    grad_eager = tape_eager.gradient(result_eager, sentinel)
    
    print(f" Eager success. Result: {result_eager.numpy()}, Grad: {grad_eager.numpy()}")

    # 2. Compiled Execution (tf.function)
    compiled_program = tf.function(fuzzed_program)
    
    with tf.GradientTape() as tape_compiled:
        result_compiled = compiled_program(arg_0, sentinel)
    grad_compiled = tape_compiled.gradient(result_compiled, sentinel)
    
    print(f" Compile success. Result: {result_compiled.numpy()}, Grad: {grad_compiled.numpy()}")

    # 3. Assertion for Divergence
    # Check if results match
    assert np.allclose(result_eager.numpy(), result_compiled.numpy()), \
        f"Result divergence detected! Eager: {result_eager.numpy()}, Compiled: {result_compiled.numpy()}"
    
    # Check if gradients match
    assert np.allclose(grad_eager.numpy(), grad_compiled.numpy()), \
        f"Gradient divergence detected! Eager: {grad_eager.numpy()}, Compiled: {grad_compiled.numpy()}"

if __name__ == "__main__":
    test_tf_math_add_compile_divergence()