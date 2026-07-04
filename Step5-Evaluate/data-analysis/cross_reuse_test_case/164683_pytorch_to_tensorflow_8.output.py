import sys
import torch

# Handle environment incompatibility (e.g., missing GLIBCXX_3.4.29 for libstdc++)
try:
    import tensorflow as tf
except ImportError as e:
    print(f"Skipping test: TensorFlow import failed due to environment issues.")
    print(f"Error details: {e}")
    sys.exit(0)

# Configure mixed precision to match the context of the original bug (bf16 usage)
# This helps in reproducing potential type casting issues similar to the PyTorch bug
policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
tf.keras.mixed_precision.set_global_policy(policy)

def foo_tf(arg0, sentinel):
    # arg0: int64 indices, shape (4, 4)
    # sentinel: bf16 scalar
    
    # 1. Use the Similar API: tf.keras.initializers.LecunNormal
    # This replaces the data generation part of the original test case.
    # We initialize the weights that would later be used in embedding/activations.
    initializer = tf.keras.initializers.LecunNormal()
    
    # Initialize weights with shape (5000, 4) and dtype bfloat16
    # Original: t5 = torch.rand([5000, 4], dtype=torch.bfloat16, ...)
    t_weights = initializer(shape=(5000, 4), dtype=tf.bfloat16)

    # 2. Apply activations (Relu -> Silu) as in the original bug
    # Original: t6 = relu(t5), t7 = silu(t6)
    t_weights = tf.nn.relu(t_weights)
    # tf.keras.activations.silu is equivalent to torch.nn.functional.silu
    t_weights = tf.keras.activations.silu(t_weights)

    # 3. Embedding lookup
    # Original: t8 = embedding(clamp(t4...), t7)
    # We use arg0 directly as indices (clamped to valid range)
    clamped_indices = tf.clip_by_value(arg0, 0, tf.shape(t_weights)[0] - 1)
    t_embedded = tf.nn.embedding_lookup(t_weights, clamped_indices)

    # 4. Reduction and Sentinel addition
    # Original: t9 = t8.min(), output = t9 + sentinel
    t_min = tf.reduce_min(t_embedded)
    output = t_min + sentinel

    return output

# Inputs setup matching the original PyTorch test case
# Original: arg0 = torch.randint(0, 1000, [4, 4], dtype=torch.int64, device='cuda')
arg0 = tf.random.uniform((4, 4), minval=0, maxval=1000, dtype=tf.int64)
# Original: sentinel = torch.tensor(0.0, dtype=torch.bfloat16, device='cuda', requires_grad=True)
sentinel = tf.constant(0.0, dtype=tf.bfloat16)

if __name__ == '__main__':
    print("Running Eager Mode...")
    try:
        out_eager = foo_tf(arg0, sentinel)
        print(f'Eager Success!  Output: {out_eager.numpy()}')
    except Exception as e:
        print(f'Eager Failed!  Error: {e}')
        sys.exit(1)

    print("\nRunning Compiled Mode (tf.function)...")
    # Using tf.function with jit_compile=True to mimic torch.compile behavior
    compiled_foo = tf.function(jit_compile=True)(foo_tf)
    try:
        out_compiled = compiled_foo(arg0, sentinel)
        print(f'Compile Success!  Output: {out_compiled.numpy()}')

        # Verify divergence (Eager vs Compile)
        # The original bug was a divergence or crash in compile mode.
        if not tf.reduce_all(tf.abs(out_eager - out_compiled) < 1e-5).numpy():
            print("Divergence detected between Eager and Compiled modes! ")
        else:
            print("No divergence detected. ")
    except Exception as e:
        print(f'Compile Failed!  Error: {e}')