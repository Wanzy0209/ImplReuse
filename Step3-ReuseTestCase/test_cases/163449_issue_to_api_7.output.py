import torch
import tensorflow as tf
import numpy as np
import sys

def test_dropout_eager_compile_divergence():
    """
    Test case for tf.nn.dropout inspired by PyTorch Issue 163449.
    
    The original issue highlights a numerical divergence between eager and compiled 
    modes when using mixed precision (bfloat16) and complex reduction chains (norm, pow).
    
    This test adapts that scenario to TensorFlow:
    1. Enables mixed precision (bfloat16) to match the sensitivity of the original bug.
    2. Uses tf.nn.dropout (the similar API) within a computation graph.
    3. Compares the output of eager execution vs. XLA compiled execution (tf.function).
    4. Includes reduction and power operations to amplify potential precision errors.
    """
    
    # Enable mixed precision policy to mimic the bfloat16/float32 interplay in the original issue
    # Note: This requires GPU support in the runtime environment
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
        tf.keras.mixed_precision.set_global_policy(policy)
    except ValueError:
        print("Warning: mixed_bfloat16 not supported on this device, falling back to float32.")
        policy = tf.keras.mixed_precision.Policy('float32')
        tf.keras.mixed_precision.set_global_policy(policy)

    # Set seeds for reproducibility, crucial for stochastic ops like dropout
    tf.random.set_seed(42)
    np.random.seed(42)

    # Input tensor mimicking the shape and type from the original issue (arg1)
    # Original: size=(5, 1024), dtype=bfloat16
    x = tf.random.normal([5, 1024], dtype=tf.float32)

    # Define the computation function
    # Original logic: addmm -> norm -> var -> pow
    # Adapted logic: dropout -> norm -> pow
    def computation_graph(x_input):
        # Apply dropout. 
        # The similar API implementation shows scaling (1 / keep_prob) and casting.
        # We use a fixed seed to ensure deterministic behavior for comparison.
        rate = 0.1
        dropped = tf.nn.dropout(x_input, rate=rate, seed=42)
        
        # Mimic the reduction chain from the bug report (t3.norm() -> t4)
        norm_val = tf.norm(dropped)
        
        # Mimic the scalar addition and power operations (t7 + t4 + t9 -> pow)
        # These operations often expose precision drift in lower precision formats.
        scalar_add = tf.constant(0.5, dtype=tf.float32)
        combined = norm_val + scalar_add
        
        # Final power operation
        result = tf.pow(norm_val, combined)
        return result

    # 1. Eager Execution
    out_eager = computation_graph(x)

    # 2. Compiled Execution (mimics torch.compile)
    # Using jit_compile=True to trigger the XLA compiler
    compiled_fn = tf.function(computation_graph, jit_compile=True)
    out_compiled = compiled_fn(x)

    # Compare outputs
    # We check for relative difference similar to the original bug report
    diff = tf.abs(out_eager - out_compiled)
    rel_diff = diff / (tf.abs(out_eager) + 1e-12)

    print(f'Eager Result: {out_eager.numpy()}')
    print(f'Compiled Result: {out_compiled.numpy()}')
    print(f'Absolute diff: {diff.numpy()}')
    print(f'Relative diff: {rel_diff.numpy()}')

    # The original bug reported > 8% difference.
    # With proper seed handling and mixed precision, we expect minimal difference.
    # If this assertion fails, it indicates a divergence similar to the reported issue.
    if rel_diff > 0.05: # 5% threshold
        print(f' Eager and Compile outputs differ significantly!')
        print(f'Relative diff: {rel_diff.numpy()}')
        sys.exit(1)
    else:
        print(' Eager and Compile outputs are consistent.')

if __name__ == '__main__':
    test_dropout_eager_compile_divergence()