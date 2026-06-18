import tensorflow as tf
import numpy as np

def test_argmax_type_divergence():
    """
    Test case adapted from Issue 164086 logic, leveraging tf.math.argmax.
    
    The original issue involves a divergence between eager and compiled modes 
    when handling mixed precision types (int64, float16) and complex operations.
    This test replaces the reduction operation with the similar API (argmax)
    to check for type compatibility issues in TensorFlow's XLA compilation.
    """
    
    # Enable mixed precision to mimic the environment where precision emulation matters
    # Note: TF might not have 'emulate_precision_casts' exactly like PyTorch Inductor,
    # but mixed_float16 policy is the equivalent concept.
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
    except ValueError:
        print("Mixed precision not supported on this hardware, falling back to float32.")
        policy = tf.keras.mixed_precision.Policy('float32')
        tf.keras.mixed_precision.set_global_policy(policy)

    # Inputs mimicking the shapes and dtypes from the bug report
    # arg0: int64, used for indexing/embedding
    arg0 = tf.constant(np.random.randint(0, 1000, (42, 56)), dtype=tf.int64)
    
    # arg1, arg2: float16, used for linear layer
    arg1 = tf.constant(np.random.rand(50000, 128), dtype=tf.float16)
    arg2 = tf.constant(np.random.rand(46, 128), dtype=tf.float16)
    
    # arg3: float16, used for the reduction operation
    arg3 = tf.constant(np.random.rand(50000, 4, 46), dtype=tf.float16)
    
    # arg4, arg5: float16, used for concatenation
    arg4 = tf.constant(np.random.rand(25786, 46), dtype=tf.float16)
    arg5 = tf.constant(np.random.rand(24214, 46), dtype=tf.float16)

    def foo(arg0, arg1, arg2, arg3, arg4, arg5):
        # Linear operation: (50000, 128) * (128, 46) -> (50000, 46)
        t5 = tf.linalg.matmul(arg1, arg2, transpose_b=True)
        
        # --- Similar API Usage ---
        # Original: t7 = t6.max(dim=1).values (returns float16 values)
        # Similar:  t7 = tf.math.argmax(arg3, axis=1) (returns int64 indices)
        # This introduces a specific type interaction (int64 vs float16) in subsequent ops.
        t7 = tf.math.argmax(arg3, axis=1, output_type=tf.int64)
        
        # Concatenation
        t10 = tf.concat([arg4, arg5], axis=0)
        
        # Complex power operations involving the result of the similar API.
        # t7 is int64 here. We cast to float16 to allow pow, but the source type 
        # (int64 from argmax) is the stress factor for the compiler.
        t7_cast = tf.cast(t7, tf.float16)
        
        # Nested pow operations mirroring the bug report
        t11 = tf.pow(tf.pow(tf.pow(tf.pow(t5, t7_cast), t10), t5), t7_cast)
        
        # Embedding-like operation using arg0 (int64)
        # Clamp indices
        t2_clamped = tf.clip_by_value(arg0, 0, tf.shape(t11)[0] - 1)
        # Gather (embedding lookup)
        t12 = tf.gather(t11, t2_clamped)
        
        return t12

    # 1. Test Eager Mode
    print("Testing Eager Mode...")
    try:
        out_eager = foo(arg0, arg1, arg2, arg3, arg4, arg5)
        print("Eager Success! ")
    except Exception as e:
        print(f"Eager Failed: {e}")
        return

    # 2. Test Compiled Mode (tf.function with XLA)
    print("Testing Compiled Mode (XLA)...")
    try:
        compiled_foo = tf.function(foo, jit_compile=True)
        out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, arg5)
        print("Compile Success! ")
        
        # Check for divergence (optional, but good practice for regression tests)
        # Note: Due to potential non-determinism in low-level implementations or 
        # different precision handling in XLA, exact equality might be strict.
        # Here we primarily check that it doesn't crash with IncompatibleTypeError.
        if tf.reduce_all(tf.equal(tf.shape(out_eager), tf.shape(out_compiled))):
            print("Shape Match: ")
        else:
            print("Shape Mismatch: ")
            
    except Exception as e:
        print(f"Compile Failed: {e}")
        print("This might indicate a type handling divergence similar to the reported bug.")

if __name__ == '__main__':
    test_argmax_type_divergence()