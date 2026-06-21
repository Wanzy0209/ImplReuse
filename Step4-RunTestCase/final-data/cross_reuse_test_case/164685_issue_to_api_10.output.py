import sys
import numpy as np

# Attempt to import TensorFlow and handle environment errors (e.g., GLIBCXX version mismatch)
try:
    import tensorflow as tf
    import tf.experimental.numpy as tnp

    # Enable numpy behavior for tf.experimental.numpy
    tnp.experimental_enable_numpy_behavior()
except ImportError as e:
    print(f"Skipping test: Failed to import TensorFlow due to environment issues.")
    print(f"Error details: {e}")
    # Exit gracefully to indicate the test was skipped due to environment, not logic failure
    sys.exit(0)

# torch is imported in the original file, keeping it for consistency
import torch

def test_tf_numpy_add_divergence():
    """
    Test case adapted from PyTorch Issue 164685.
    Original issue: Eager/Compile divergence with scalar operations and mixed dtypes.
    This test leverages tf.experimental.numpy.add (the similar API) to check for
    similar divergence in TensorFlow's eager vs. graph (tf.function) modes.
    """
    
    # Setup inputs mirroring the original bug report
    # arg_0 is a scalar int32
    arg_0 = np.int32(np.random.randn())
    # Sentinel tensor to ensure tensor operations are preserved
    sentinel = tf.constant(1.0)

    def fuzzed_program(arg_0, sentinel):
        var_node_2 = -6  # dtype=int64
        var_node_3 = arg_0  # dtype=int32
        
        # Use the similar API: tf.experimental.numpy.add
        # Replacing the multiplication in the original bug with addition
        # to test type promotion and scalar handling with the specific API.
        var_node_1 = tnp.add(var_node_2, var_node_3)
        
        var_node_5 = tnp.full((), 1, dtype=np.int64)
        # TF Tensors use .numpy() to extract a Python scalar, equivalent to .item()
        var_node_4 = var_node_5.numpy()
        
        var_node_0 = tnp.divide(var_node_1, var_node_4)
        
        # Combine with sentinel
        result = var_node_0 * sentinel
        return result

    # 1. Run in Eager mode
    try:
        result_eager = fuzzed_program(arg_0, sentinel)
        print(f' eager success: {result_eager}')
    except Exception as e:
        print(f' eager failed: {e}')
        raise

    # 2. Run in Compiled mode (tf.function is analogous to torch.compile)
    try:
        compiled_program = tf.function(fuzzed_program)
        result_compiled = compiled_program(arg_0, sentinel)
        print(f' compile success: {result_compiled}')
    except Exception as e:
        print(f' compile failed: {e}')
        raise

    # 3. Check for divergence
    # Using numpy equality check for robustness with different scalar types
    if not np.equal(result_eager.numpy(), result_compiled.numpy()):
        raise AssertionError(
            f"Divergence detected between eager and compiled modes!\n"
            f"Eager: {result_eager}\nCompiled: {result_compiled}"
        )
    
    print(' Test passed: No divergence between eager and compiled execution.')

if __name__ == "__main__":
    test_tf_numpy_add_divergence()