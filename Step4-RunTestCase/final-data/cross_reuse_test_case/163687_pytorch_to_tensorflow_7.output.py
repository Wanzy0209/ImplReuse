import tensorflow as tf
import numpy as np

def flex_attention_sim(q, k, v):
    """
    Simulates the behavior of flex_attention using standard TensorFlow operations.
    Performs scaled dot-product attention: softmax(Q @ K.T) @ V.
    """
    # q, k, v shapes: (Batch, Heads, Seq, Dim)
    # Matmul for Q and K: (Batch, Heads, Seq, Seq)
    attn_weights = tf.matmul(q, k, transpose_b=True)
    attn_weights = tf.nn.softmax(attn_weights)
    # Matmul for weights and V: (Batch, Heads, Seq, Dim)
    output = tf.matmul(attn_weights, v)
    return output

def foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10):
    """
    Replicates the logic from the original PyTorch bug report.
    Uses tf.keras.backend.name_scope to group operations, testing behavior
    in eager vs compiled modes.
    """
    # Using the similar API: tf.keras.backend.name_scope
    with tf.keras.backend.name_scope("flex_attention_block"):
        t0 = arg0
        t1 = arg1
        t2 = arg2
        t3 = flex_attention_sim(t0, t1, t2)

        t4 = arg3
        t5 = arg4
        t6 = arg5
        t7 = flex_attention_sim(t4, t5, t6)

        t8 = flex_attention_sim(t3, t7, t7)

        t9 = arg6
        t10 = arg7
        t11 = flex_attention_sim(t9, t7, t10)

        t12 = flex_attention_sim(t11, t8, t3)

        t13 = arg8
        t14 = arg9
        t15 = flex_attention_sim(t13, t2, t14)

        t16 = arg10
        # t17 = t16.clone(); t17.zero_()
        t17 = tf.zeros_like(t16)

        t18 = flex_attention_sim(t17, t8, t3)

        t19 = flex_attention_sim(t15, t17, t18)

        t20 = flex_attention_sim(t8, t12, t19)

        output = t20
    return output

# Define tensor shapes based on the original bug report
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

# Create random tensors
args = [tf.random.normal(shape, dtype=tf.float32) for shape in shapes]

# Test Case 1: Eager Execution
print("Testing Eager Execution...")
try:
    eager_output = foo(*args)
    print(f"Eager Execution Successful. Output shape: {eager_output.shape}")
except Exception as e:
    print(f"Eager Execution Failed: {e}")

# Test Case 2: Compiled Execution (tf.function)
# This corresponds to the 'compile' mode in the original PyTorch issue
print("\nTesting Compiled Execution...")
compiled_foo = tf.function(foo)
try:
    compiled_output = compiled_foo(*args)
    print(f"Compiled Execution Successful. Output shape: {compiled_output.shape}")
except Exception as e:
    print(f"Compiled Execution Failed: {e}")

# Verification: Check for shape consistency between modes
if 'eager_output' in locals() and 'compiled_output' in locals():
    assert eager_output.shape == compiled_output.shape, \
        f"Shape divergence: Eager {eager_output.shape} vs Compiled {compiled_output.shape}"
    print("Verification Passed: Shapes match between Eager and Compiled modes.")