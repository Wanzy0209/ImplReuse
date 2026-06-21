try:
    import torch
    import tensorflow as tf
    from tensorflow.keras import mixed_precision
    import numpy as np
except ImportError as e:
    print(f"Skipping test due to missing dependencies or environment incompatibility: {e}")
    print("This is likely due to a system library mismatch (e.g., GLIBCXX version).")
    exit(0)

# Configure mixed precision to match the float16 context of the original bug
# The original bug involved emulate_precision_casts and float16 tensors
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

def test_argmax_mixed_precision_divergence():
    """
    Test case adapted from Issue 164086.
    Original bug: IncompatibleTypeErrorImpl in PyTorch when using torch.tanh on int64
    within a complex float16 graph under torch.compile.
    
    Adaptation: Uses tf.keras.backend.argmax (Similar API) to generate int64 indices
    within a complex float16 graph to test type handling and compilation consistency.
    """
    
    # Inputs mirroring the original bug report
    # t0 equivalent: Used to generate indices via argmax
    # Shape (42, 56) -> we add a dim to argmax over, matching the dimensionality of t0
    x_input = tf.random.uniform([42, 56, 10], dtype=tf.float16)

    # t3, t4 equivalent for Linear (MatMul)
    w_input = tf.random.uniform([50000, 128], dtype=tf.float16)
    linear_weights = tf.random.uniform([46, 128], dtype=tf.float16)

    # t6 equivalent for Max
    max_input = tf.random.uniform([50000, 4, 46], dtype=tf.float16)

    # t8, t9 equivalent for Cat
    cat_1 = tf.random.uniform([25786, 46], dtype=tf.float16)
    cat_2 = tf.random.uniform([24214, 46], dtype=tf.float16)

    # 1. Generate indices using the Similar API (tf.keras.backend.argmax)
    # This replaces the torch.tanh(t0) logic which produced an int64-like tensor
    # used for embedding lookup. argmax naturally returns int64.
    indices = tf.keras.backend.argmax(x_input, axis=-1) # Shape (42, 56), dtype int64

    # 2. Linear operation (torch.nn.functional.linear)
    t5 = tf.linalg.matmul(w_input, linear_weights, transpose_b=True)

    # 3. Max operation (t6.max(dim=1).values)
    t7 = tf.reduce_max(max_input, axis=1)

    # 4. Concatenation (torch.cat)
    t10 = tf.concat([cat_1, cat_2], axis=0)

    # 5. Complex Pow operations (sensitive to precision)
    # t11 = pow(pow(pow(pow(t5, t7), t10), t5), t7)
    # Note: tf.pow supports float16, but complex chains can trigger precision issues
    t11 = tf.pow(tf.pow(tf.pow(tf.pow(t5, t7), t10), t5), t7)

    # 6. Embedding Lookup using indices from argmax
    # This mirrors the torch.nn.functional.embedding(t2, t11) line
    # Clamp indices to ensure they are within bounds of the embedding table (t11)
    clamped_indices = tf.clip_by_value(indices, 0, tf.shape(t11)[0] - 1)
    t12 = tf.nn.embedding_lookup(t11, clamped_indices)

    # 7. Output with sentinel
    sentinel = tf.constant(0.0, dtype=tf.float16)
    output = t12 + sentinel
    return output

if __name__ == '__main__':
    print("Running test for Issue 164086 adaptation (tf.keras.backend.argmax)...")

    # Test Eager Execution
    try:
        out_eager = test_argmax_mixed_precision_divergence()
        print("Eager Execution Success! ")
    except Exception as e:
        print(f"Eager Execution Failed: {e}")
        out_eager = None

    # Test Compiled Execution (tf.function)
    # This mirrors the torch.compile check in the original bug report
    try:
        compiled_func = tf.function(test_argmax_mixed_precision_divergence)
        out_compiled = compiled_func()
        print("Compiled Execution Success! ")

        # Check for divergence between Eager and Compiled modes
        if out_eager is not None:
            # Using a slightly higher tolerance for float16 operations
            if tf.reduce_all(tf.abs(out_eager - out_compiled) < 1e-2).numpy():
                print("Consistency Check Passed! ")
            else:
                print("Consistency Check Failed!  (Divergence detected between Eager and Compiled)")
                print(f"Max diff: {tf.reduce_max(tf.abs(out_eager - out_compiled)).numpy()}")
    except Exception as e:
        print(f"Compiled Execution Failed: {e}")