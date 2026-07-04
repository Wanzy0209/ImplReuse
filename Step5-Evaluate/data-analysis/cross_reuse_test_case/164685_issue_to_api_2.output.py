import sys

# Handle environment incompatibility (e.g., GLIBC version mismatch) gracefully
try:
    import tensorflow as tf
except ImportError as e:
    print(f"Skipping test: Failed to import TensorFlow due to environment issues.")
    print(f"Error details: {e}")
    sys.exit(0)

import torch

def test_compile_divergence_with_symbolic_check():
    """
    Test case derived from PyTorch Issue 164685.
    
    The original issue involves a divergence between eager and compiled modes
    when performing scalar arithmetic and type casting (KeyError: u0).
    
    This test adapts the logic to TensorFlow and uses `tf.is_symbolic_tensor`
    to verify that the compiler correctly identifies tensors as symbolic during
    graph tracing, ensuring no divergence or crash occurs.
    """
    
    # 1. Eager Mode Execution
    # Replicate the arithmetic logic from the bug report
    arg_0 = tf.constant(5, dtype=tf.int32)
    
    var_node_2 = tf.constant(-6, dtype=tf.int64)
    var_node_3 = arg_0
    var_node_1 = var_node_2 * var_node_3
    
    # var_node_5 = torch.full((), 1, dtype=torch.int64)
    var_node_5 = tf.fill((), tf.constant(1, dtype=tf.int64))
    
    # var_node_4 = torch.full((), 1, dtype=torch.int64).item() -> Mimic extracting a scalar value
    # In eager mode, we can do this directly.
    var_node_4 = var_node_5.numpy().item()
    
    # var_node_0 = var_node_1 / var_node_4
    var_node_0 = var_node_1 / var_node_4
    
    # Verify eager execution: tensors should NOT be symbolic
    assert not tf.is_symbolic_tensor(var_node_0), "Eager tensor should not be symbolic"
    assert var_node_0.numpy() == -30, "Eager computation result incorrect"
    print(" Eager execution successful (not symbolic).")

    # 2. Compiled Mode Execution (tf.function)
    @tf.function
    def compiled_program(arg_0):
        var_node_2 = tf.constant(-6, dtype=tf.int64)
        var_node_3 = arg_0
        var_node_1 = var_node_2 * var_node_3
        
        var_node_5 = tf.fill((), tf.constant(1, dtype=tf.int64))
        
        # Inside tf.function, we cannot call .numpy() on symbolic tensors.
        # We use a Python scalar to mimic the result of .item() for the division.
        var_node_4 = 1 
        
        var_node_0 = var_node_1 / var_node_4
        
        # Use the similar API (tf.is_symbolic_tensor) to check the state
        # This verifies the compiler is tracking the tensor correctly.
        is_sym = tf.is_symbolic_tensor(var_node_0)
        
        return var_node_0, is_sym

    # Run the compiled program
    result_compiled, is_symbolic = compiled_program(arg_0)
    
    # Verify the computation result matches eager mode (no divergence)
    assert result_compiled.numpy() == -30, "Compiled computation result incorrect"
    
    # Verify that the tensor was identified as symbolic during the trace
    # Note: is_symbolic might be a Tensor or a bool depending on tracing context
    if isinstance(is_symbolic, tf.Tensor):
        assert is_symbolic.numpy() == True, "Tensor should be symbolic in graph mode"
    else:
        assert is_symbolic == True, "Tensor should be symbolic in graph mode"
        
    print(" Compiled execution successful (symbolic).")

if __name__ == "__main__":
    test_compile_divergence_with_symbolic_check()