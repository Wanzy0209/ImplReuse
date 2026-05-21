import tensorflow as tf
import numpy as np
import sys

# Helper function to mimic PyTorch's flex_attention using TensorFlow primitives
# Implements Scaled Dot-Product Attention
def flex_attention(q, k, v):
    # Inputs are assumed to be shape (Batch, Heads, Seq, Dim)
    dim = tf.cast(tf.shape(q)[-1], tf.float32)
    scale = tf.math.rsqrt(dim)
    
    # Q * K^T
    logits = tf.matmul(q, k, transpose_b=True)
    logits *= scale
    
    # Softmax
    weights = tf.nn.softmax(logits, axis=-1)
    
    # Weights * V
    return tf.matmul(weights, v)

# The main computation function adapted from the PyTorch 'foo'
def computation_fn(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10):
    t0 = arg0
    t1 = arg1
    t2 = arg2
    t3 = flex_attention(t0, t1, t2)

    t4 = arg3
    t5 = arg4
    t6 = arg5
    t7 = flex_attention(t4, t5, t6)

    t8 = flex_attention(t3, t7, t7)

    t9 = arg6
    t10 = arg7
    t11 = flex_attention(t9, t7, t10)

    t12 = flex_attention(t11, t8, t3)

    t13 = arg8
    t14 = arg9
    t15 = flex_attention(t13, t2, t14)

    t16 = arg10
    # PyTorch: t17 = t16.clone(); t17.zero_()
    # TensorFlow equivalent: create a tensor of zeros with the same properties
    t17 = tf.zeros_like(t16)

    t18 = flex_attention(t17, t8, t3)
    t19 = flex_attention(t15, t17, t18)
    t20 = flex_attention(t8, t12, t19)

    return t20

def run_test():
    # Define shapes based on the bug report
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

    # Create inputs
    inputs = [tf.random.normal(shape, dtype=tf.float32) for shape in shapes]

    # 1. Test Eager Execution
    print("Testing Eager Execution...")
    try:
        eager_output = computation_fn(*inputs)
        print(f"Eager Success. Output shape: {eager_output.shape}")
    except Exception as e:
        print(f"Eager Failed: {e}")
        return

    # 2. Test TPU Rewrite (Compiled)
    # Note: This requires a TPU environment. 
    # We attempt to initialize TPU, but handle the case where it's not available.
    print("Testing tf.compat.v1.tpu.rewrite (Compiled)...")
    
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        
        # tf.compat.v1.tpu.rewrite compiles the function for TPU
        # It returns a list of tensors representing the result of the computation.
        compiled_op = tf.compat.v1.tpu.rewrite(computation_fn, inputs)
        
        # To execute the compiled op, we typically use a session in TF1 compat mode
        # or evaluate it within the TPU strategy context in TF2.
        # Since rewrite is a compat.v1 API, we use a Session.
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            result = sess.run(compiled_op)
            
            # result is a list of outputs from the function
            compiled_output = result[0]
            print(f"Compiled Success. Output shape: {compiled_output.shape}")
            
            # Verify behavior: Check if shapes match (basic sanity check)
            assert eager_output.shape == compiled_output.shape, "Shape mismatch between eager and compiled"
            print("Test Passed: Eager and Compiled outputs have matching shapes.")

    except tf.errors.NotFoundError:
        print("TPU device not found. Skipping compiled execution test.")
    except Exception as e:
        print(f"Compiled Execution Failed (Potential OOM or Divergence): {e}")
        # In the context of the bug report, an OOM here would be the reproduction.

if __name__ == "__main__":
    run_test()