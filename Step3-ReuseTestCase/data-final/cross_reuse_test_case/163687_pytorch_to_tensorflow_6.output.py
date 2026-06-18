import torch
import tensorflow as tf
import numpy as np

# Ensure we are using TensorFlow 2.x behavior but can access v1 APIs
tf.compat.v1.disable_eager_execution() 
# Note: We disable eager execution globally to strictly test graph mode behavior 
# similar to the "compile" scenario in the bug report, but we will use tf.function 
# to explicitly test the divergence if eager was enabled. 
# However, to strictly follow the "Cross-library" adaptation where the original 
# bug was "Eager vs Compile", we will enable eager and use tf.function for the compile part.

import tensorflow.compat.v1 as tfv1
tfv1.enable_eager_execution()

def flex_attention_tf(q, k, v):
    """
    TensorFlow equivalent of the flex_attention operation used in the bug report.
    Performs standard scaled dot-product attention.
    Shapes:
        q: (Batch, Heads, Seq_Q, Dim)
        k: (Batch, Heads, Seq_K, Dim)
        v: (Batch, Heads, Seq_K, Dim_V)
    Returns:
        (Batch, Heads, Seq_Q, Dim_V)
    """
    # Transpose k to (Batch, Heads, Dim, Seq_K) for matmul
    kt = tf.transpose(k, [0, 1, 3, 2])
    
    # Matmul: (Batch, Heads, Seq_Q, Seq_K)
    scores = tf.matmul(q, kt)
    
    # Softmax
    weights = tf.nn.softmax(scores, axis=-1)
    
    # Matmul with v: (Batch, Heads, Seq_Q, Dim_V)
    output = tf.matmul(weights, v)
    return output

def run_model(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10):
    """
    The core logic from the bug report, adapted to TensorFlow.
    Wrapped in tf.compat.v1.name_scope as requested.
    """
    # Using the requested API: tf.compat.v1.name_scope
    with tf.compat.v1.name_scope("flex_attention_block"):
        t0 = arg0
        t1 = arg1
        t2 = arg2
        t3 = flex_attention_tf(t0, t1, t2)

        t4 = arg3
        t5 = arg4
        t6 = arg5
        t7 = flex_attention_tf(t4, t5, t6)

        t8 = flex_attention_tf(t3, t7, t7)

        t9 = arg6
        t10 = arg7
        t11 = flex_attention_tf(t9, t7, t10)

        t12 = flex_attention_tf(t11, t8, t3)

        t13 = arg8
        t14 = arg9
        t15 = flex_attention_tf(t13, t2, t14)

        t16 = arg10
        # Equivalent of t16.clone(); t17.zero_()
        t17 = tf.zeros_like(t16) 
        t18 = flex_attention_tf(t17, t8, t3)

        t19 = flex_attention_tf(t15, t17, t18)

        t20 = flex_attention_tf(t8, t12, t19)
        
        return t20

# Generate inputs matching the shapes from the PyTorch bug report
# Shapes: (27, 26, X, 122)
shapes = [
    [27, 26, 62, 122],   # arg0
    [27, 26, 124, 122],  # arg1
    [27, 26, 124, 122],  # arg2
    [27, 26, 124, 122],  # arg3
    [27, 26, 248, 122],  # arg4
    [27, 26, 248, 122],  # arg5
    [27, 26, 31, 122],   # arg6
    [27, 26, 124, 122],  # arg7
    [27, 26, 31, 122],   # arg8
    [27, 26, 124, 122],  # arg9
    [27, 26, 124, 122]   # arg10
]

inputs = [tf.random.normal(shape, dtype=tf.float32) for shape in shapes]

print("Testing Eager Execution...")
try:
    # Run in eager mode
    output_eager = run_model(*inputs)
    print(f"Eager Execution Success. Output shape: {output_eager.shape}")
except Exception as e:
    print(f"Eager Execution Failed: {e}")

print("\nTesting Compiled Execution (tf.function)...")
try:
    # Run in compiled mode (Graph mode) - analogous to torch.compile
    compiled_model = tf.function(run_model)
    output_compiled = compiled_model(*inputs)
    print(f"Compiled Execution Success. Output shape: {output_compiled.shape}")
except Exception as e:
    print(f"Compiled Execution Failed: {e}")

# Verify consistency if both succeeded
if 'output_eager' in locals() and 'output_compiled' in locals():
    assert output_eager.shape == output_compiled.shape, \
        f"Shape divergence: Eager {output_eager.shape} vs Compiled {output_compiled.shape}"
    print("Assertion Passed: Eager and Compiled output shapes match.")