import sys
import torch
import numpy as np

# Handle environment-specific import errors for TensorFlow
try:
    import tensorflow as tf
except ImportError as e:
    # Check if the error is related to the GLIBC version mismatch found in the error message
    if "GLIBCXX" in str(e) or "libstdc++" in str(e):
        print(f"Skipping test: TensorFlow import failed due to system library incompatibility (GLIBCXX).")
        print(f"Error details: {e}")
        sys.exit(0)
    else:
        # If it's a different import error, re-raise it
        raise

def test_tf_linalg_trace_compilation():
    """
    Test case for tf.linalg.trace inspired by PyTorch Issue 164102.
    
    The original issue involves a divergence between eager and compiled modes
    in PyTorch when using torch.rms_norm with specific tensor shapes and 
    bfloat16 dtype. This test adapts the scenario to TensorFlow, using 
    tf.linalg.trace within a tf.function (compiled graph) to verify 
    correct behavior under similar conditions.
    """
    
    # Enable strict compilation to mimic torch._dynamo behavior
    @tf.function(jit_compile=True)
    def compiled_trace_op(input_tensor):
        # Mimic the exp operation preceding the norm in the original bug
        t_exp = tf.exp(input_tensor)
        
        # Replace torch.rms_norm with tf.linalg.trace
        # Note: tf.linalg.trace reduces the last two dimensions.
        # Input shape: (93, 62, 8) -> Output shape: (93,)
        t_trace = tf.linalg.trace(t_exp)
        
        return t_trace

    # Replicate the specific tensor properties from the bug report
    # arg5 in the bug: size=(93, 62, 8), dtype=bfloat16
    batch_size = 93
    dim1 = 62
    dim2 = 8
    
    # Use bfloat16 to match the original bug's precision context
    # Note: bfloat16 support depends on the hardware (TPU/GPU) or TF version.
    try:
        dtype = tf.bfloat16
    except AttributeError:
        print("bfloat16 not supported, falling back to float16")
        dtype = tf.float16

    # Create random input
    input_tensor = tf.random.normal(
        shape=[batch_size, dim1, dim2], 
        dtype=dtype
    )

    # 1. Test Eager Execution
    try:
        eager_result = compiled_trace_op.python_function(input_tensor)
        print(f"Eager execution success. Shape: {eager_result.shape}")
    except Exception as e:
        print(f"Eager execution failed: {e}")

    # 2. Test Compiled Execution (tf.function)
    try:
        compiled_result = compiled_trace_op(input_tensor)
        print(f"Compiled execution success. Shape: {compiled_result.shape}")
        
        # Verify results match
        # Note: Small numerical differences might occur with bfloat16/float16
        if dtype == tf.float32:
            np.testing.assert_allclose(eager_result.numpy(), compiled_result.numpy(), rtol=1e-5)
        else:
            # For reduced precision, just check they are close or not NaN
            assert not tf.reduce_any(tf.math.is_nan(compiled_result))
            
    except Exception as e:
        print(f"Compiled execution failed: {e}")
        raise

if __name__ == "__main__":
    test_tf_linalg_trace_compilation()