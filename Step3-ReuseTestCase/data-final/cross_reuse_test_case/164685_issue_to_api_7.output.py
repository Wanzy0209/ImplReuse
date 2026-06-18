import torch
import tensorflow as tf
import numpy as np

def test_truncated_normal_compile_divergence():
    """
    Test case derived from PyTorch Issue 164685.
    This test checks for eager vs. graph compilation divergence when using
    tf.keras.initializers.TruncatedNormal to generate scalar inputs that are
    then subjected to specific type casting and arithmetic operations.
    """
    # Set seed for reproducibility
    tf.random.set_seed(19989)

    # The similar API: TruncatedNormal
    # Used here to generate the initial scalar value, mirroring torch.randn(())
    initializer = tf.keras.initializers.TruncatedNormal(
        mean=0.0, stddev=1.0, seed=19989, dtype=tf.float32
    )

    def fuzzed_program():
        # Generate a scalar value
        raw_val = initializer(shape=())
        
        # Replicate the logic from the PyTorch bug report:
        # var_node_3 = arg_0 (dtype=int32)
        var_node_3 = tf.cast(raw_val, tf.int32)
        
        # var_node_2 = -6 (dtype=int64)
        var_node_2 = tf.constant(-6, dtype=tf.int64)
        
        # var_node_1 = var_node_2 * var_node_3
        # PyTorch bug report indicates result is int32, so we cast var_node_2
        var_node_1 = tf.cast(var_node_2, tf.int32) * var_node_3
        
        # var_node_5 = torch.full((), 1, dtype=torch.int64)
        var_node_5 = tf.constant(1, dtype=tf.int64)
        
        # var_node_0 = var_node_1 / var_node_4
        # The bug report comment says dtype=int64. 
        # We cast operands to int64 to match the intended type stress test.
        var_node_0 = tf.cast(var_node_1, tf.int64) / var_node_5
        
        return var_node_0

    # 1. Run in Eager mode
    try:
        result_eager = fuzzed_program()
        print(f" Eager success: {result_eager.numpy()}")
    except Exception as e:
        print(f" Eager failure: {e}")
        return

    # 2. Run in Compiled mode (tf.function)
    # This is the TensorFlow equivalent to torch.compile
    compiled_program = tf.function(fuzzed_program)
    
    try:
        result_compiled = compiled_program()
        print(f" Compile success: {result_compiled.numpy()}")
    except Exception as e:
        print(f" Compile failure: {e}")
        raise

    # 3. Check for divergence
    # The original bug was a KeyError (crash) in compile mode.
    # Here we ensure the values match if execution succeeds.
    if result_eager.dtype.is_floating:
        # Allow for small floating point differences if promotion occurred
        assert np.isclose(result_eager.numpy(), result_compiled.numpy()), \
            "Divergence detected between eager and compiled results."
    else:
        assert tf.equal(result_eager, result_compiled).numpy(), \
            "Divergence detected between eager and compiled results."
            
    print(" Test passed: No divergence between eager and compiled execution.")

if __name__ == "__main__":
    test_truncated_normal_compile_divergence()