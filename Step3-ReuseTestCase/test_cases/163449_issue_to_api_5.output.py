import torch
import tensorflow as tf
import numpy as np
import sys

# Ensure reproducibility
tf.random.set_seed(42)

def test_dropout_eager_compile_divergence():
    """
    Test case adapted from Issue 163449 (PyTorch addmm divergence).
    
    This test checks for numerical divergence between eager and compiled (XLA) execution
    when using the tf.keras.backend.dropout pattern (scaling + masking) followed by
    reduction and power operations on bfloat16 tensors.
    """
    
    # Setup inputs similar to the original bug: bfloat16 tensors
    # Original arg1 was size (5, 1024)
    x = tf.random.uniform((5, 1024), minval=-1.0, maxval=1.0, dtype=tf.bfloat16)
    rate = 0.5
    seed = 42

    # Logic derived from the "Similar API information" (tf.keras.backend.dropout)
    # We implement this logic directly to ensure deterministic behavior (stateless random)
    # which is required to accurately compare Eager vs Compiled outputs.
    def apply_dropout_pattern(x_in, rate_in, seed_in):
        x_dtype = x_in.dtype
        keep_prob = 1.0 - rate_in
        scale = 1.0 / keep_prob
        scale = tf.cast(scale, x_dtype)
        
        # gen_math_ops.mul(x, scale)
        ret = tf.math.multiply(x_in, scale)
        
        # Generate noise mask
        noise_shape = tf.shape(x_in)
        # stateless_random_uniform requires a shape [2] seed.
        seed_pair = [seed_in, 0]
        random_tensor = tf.random.stateless_uniform(
            noise_shape, seed=seed_pair, minval=0.0, maxval=1.0, dtype=x_dtype
        )
        keep_mask = random_tensor >= rate_in
        # gen_math_ops.mul(ret, gen_math_ops.cast(keep_mask, x_dtype))
        ret = tf.math.multiply(ret, tf.cast(keep_mask, x_dtype))
        return ret

    # Computation graph mimicking the original bug's flow:
    # addmm -> norm -> pow
    # Here: dropout -> norm -> pow
    def computation_fn(x_in):
        # Apply dropout pattern
        t_dropped = apply_dropout_pattern(x_in, rate, seed)
        
        # Norm (reduction) - stays in bfloat16 initially
        t_norm = tf.norm(t_dropped)
        
        # Cast to float32 and perform power operations (mixing precision)
        # Original: t11 = torch.pow(torch.pow(t4, t7), t10)
        t_f32 = tf.cast(t_norm, tf.float32)
        
        # Create a scalar exponent to mimic t7/t10 in original bug
        exponent = tf.constant(2.0, dtype=tf.float32)
        
        # Power operation
        result = tf.pow(t_f32, exponent)
        return result

    # 1. Eager Execution
    out_eager = computation_fn(x)
    print('Eager Success! ')

    # 2. Compiled Execution (XLA)
    # Using tf.function with jit_compile=True to mimic torch.compile
    compiled_fn = tf.function(computation_fn, jit_compile=True)
    out_compiled = compiled_fn(x)
    print('Compile Success! ')

    # 3. Compare outputs
    # We expect exact match because we used stateless random with fixed seed
    diff = tf.abs(out_eager - out_compiled)
    rel_diff = diff / (tf.abs(out_eager) + 1e-12)
    
    print(f'Relative diff: {rel_diff.numpy():.6f}')
    
    # Check for significant divergence (threshold from original bug was 5%)
    if rel_diff > 0.05:
        print(f' Forward outputs differ significantly (relative)!')
        print('out_eager:', out_eager.numpy())
        print('out_compiled:', out_compiled.numpy())
        print('Absolute diff:', diff.numpy())
        print('Relative diff (%):', rel_diff.numpy() * 100)
        sys.exit(1)
    else:
        print(' Numerical outputs match within tolerance.')

if __name__ == '__main__':
    test_dropout_eager_compile_divergence()