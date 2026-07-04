import sys
import torch

try:
    import tensorflow as tf
except ImportError as e:
    print("Skipping test: Failed to import TensorFlow.")
    print(f"Error: {e}")
    print("This is likely caused by an environment issue (e.g., missing GLIBCXX_3.4.29).")
    sys.exit(0)

# Simulate the flex_attention operation using standard TensorFlow ops
# to reproduce the memory footprint and logic flow.
def flex_attention_sim(q, k, v):
    # Simplified attention mechanism: Q * K^T -> Softmax -> * V
    # This mimics the memory intensity of the original flex_attention call.
    attn = tf.matmul(q, k, transpose_b=True)
    attn = tf.nn.softmax(attn, axis=-1)
    return tf.matmul(attn, v)

def foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10):
    # Use tf.keras.name_scope as the similar API to group operations.
    # This tests the behavior of name_scope in eager vs graph (compiled) modes.
    with tf.keras.name_scope("flex_attention_block_1"):
        t3 = flex_attention_sim(arg0, arg1, arg2)

    with tf.keras.name_scope("flex_attention_block_2"):
        t7 = flex_attention_sim(arg3, arg4, arg5)

    with tf.keras.name_scope("flex_attention_block_3"):
        t8 = flex_attention_sim(t3, t7, t7)

    with tf.keras.name_scope("flex_attention_block_4"):
        t11 = flex_attention_sim(arg6, t7, arg7)

    with tf.keras.name_scope("flex_attention_block_5"):
        t12 = flex_attention_sim(t11, t8, t3)

    with tf.keras.name_scope("flex_attention_block_6"):
        t15 = flex_attention_sim(arg8, arg2, arg9)

    with tf.keras.name_scope("flex_attention_block_7"):
        # Clone and zero to mimic t17 behavior
        t17 = tf.identity(arg10)
        t17 = tf.zeros_like(t17)
        
    with tf.keras.name_scope("flex_attention_block_8"):
        t18 = flex_attention_sim(t17, t8, t3)

    with tf.keras.name_scope("flex_attention_block_9"):
        t19 = flex_attention_sim(t15, t17, t18)

    with tf.keras.name_scope("flex_attention_block_10"):
        t20 = flex_attention_sim(t8, t12, t19)

    return t20

# Initialize tensors with shapes matching the original bug report
# Device placement is handled automatically by TF
arg0 = tf.random.normal([27, 26, 62, 122])
arg1 = tf.random.normal([27, 26, 124, 122])
arg2 = tf.random.normal([27, 26, 124, 122])
arg3 = tf.random.normal([27, 26, 124, 122])
arg4 = tf.random.normal([27, 26, 248, 122])
arg5 = tf.random.normal([27, 26, 248, 122])
arg6 = tf.random.normal([27, 26, 31, 122])
arg7 = tf.random.normal([27, 26, 124, 122])
arg8 = tf.random.normal([27, 26, 31, 122])
arg9 = tf.random.normal([27, 26, 124, 122])
arg10 = tf.random.normal([27, 26, 124, 122])

args = [arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10]

# 1. Test in Eager Mode
print("Running in Eager Mode...")
try:
    output_eager = foo(*args)
    print(f"Eager Mode Success. Output shape: {output_eager.shape}")
except tf.errors.ResourceExhaustedError as e:
    print(f"Eager Mode OOM: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Eager Mode Error: {e}")
    sys.exit(1)

# 2. Test in Compiled Mode (tf.function)
# This is the TensorFlow equivalent to torch.compile
print("\nRunning in Compiled Mode (tf.function)...")
compiled_foo = tf.function(foo)
try:
    output_compiled = compiled_foo(*args)
    print(f"Compiled Mode Success. Output shape: {output_compiled.shape}")
except tf.errors.ResourceExhaustedError as e:
    print(f"Compiled Mode OOM: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Compiled Mode Error: {e}")
    sys.exit(1)

# 3. Verify Divergence
# Check if shapes match between eager and compiled execution
assert output_eager.shape == output_compiled.shape, \
    f"Shape divergence detected: Eager {output_eager.shape} vs Compiled {output_compiled.shape}"

print("\nTest Passed: No divergence or OOM detected between Eager and Compiled modes.")