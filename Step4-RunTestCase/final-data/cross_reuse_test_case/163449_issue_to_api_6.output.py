import torch
import tensorflow as tf
import numpy as np
import sys

def test_dropout_eager_compile_divergence():
    """
    Test case for tf.keras.random.dropout based on the PyTorch issue 163449.
    The original issue highlights a numerical divergence between eager and compiled modes
    involving bfloat16 and specific tensor operations.
    
    This test adapts that logic to tf.keras.random.dropout, checking for consistency
    between eager execution and XLA compilation (tf.function with jit_compile=True).
    """
    
    # Setup mixed precision to match the bfloat16 context of the original bug
    # The original bug used dtype=bfloat16 explicitly.
    policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.set_global_policy(policy)

    # Inputs mirroring the shapes in the PyTorch issue
    # PyTorch arg1: size=(5, 1024), dtype=bfloat16
    x = tf.random.uniform((5, 1024), minval=-1.0, maxval=1.0, dtype=tf.bfloat16)
    rate = 0.5
    seed = 42  # Seed is required for reproducibility in dropout

    # The function to test, mimicking the structure of the PyTorch 'foo'
    # It uses the similar API (tf.keras.random.dropout) and applies reductions
    # similar to the original bug (norm, var, pow) to stress numerical precision.
    def model_fn(x, rate, seed):
        # Similar API: tf.keras.random.dropout
        # Replaces the torch.addmm logic in terms of being the core op under test
        t_dropped = tf.keras.random.dropout(x, rate=rate, seed=seed)

        # Mimic the reduction and math chain from the PyTorch bug
        # PyTorch: t4 = t3.norm()
        t_norm = tf.norm(t_dropped)

        # PyTorch: t7 = t6.var() (where t6 was t5.var(dim=0))
        # We perform a variance calculation on the dropped tensor
        t_var = tf.math.reduce_variance(t_dropped)

        # PyTorch: t11 = torch.pow(torch.pow(t4, t7), t10)
        # We perform a power operation to check numerical stability
        # Adding a small epsilon to t_var to avoid invalid pow(0, 0) or similar edge cases
        t_pow = tf.pow(t_norm, t_var + 1e-6)

        return t_pow

    # 1. Eager Execution
    out_eager = model_fn(x, rate, seed)
    print('Eager Success! ')

    # 2. Compiled Execution (tf.function with XLA)
    # This corresponds to torch.compile in the original issue
    compiled_fn = tf.function(model_fn, jit_compile=True)
    out_compiled = compiled_fn(x, rate, seed)
    print('Compile Success! ')

    # 3. Comparison Logic (from PyTorch issue)
    # Compare outputs (forward)
    # The original bug checked relative difference of the sum.
    # Here we compare the scalar output directly.
    
    diff = tf.abs(out_eager - out_compiled).numpy()
    # Avoid division by zero
    denominator = tf.abs(out_eager).numpy() + 1e-12
    rel_diff = (diff / denominator) * 100

    print(f'Relative diff: {rel_diff:.6f}%')
    
    # The original bug failed if rel_diff > 5%.
    # For a correct implementation with a fixed seed, we expect very high precision.
    # We assert that they are close, ensuring no divergence bug exists.
    if rel_diff > 1e-5:
        print(f' Forward outputs differ significantly (relative)!')
        print('out_eager:', out_eager.numpy())
        print('out_compiled:', out_compiled.numpy())
        print('Absolute diff:', diff)
        print('Relative diff (%):', rel_diff)
        sys.exit(1)
    else:
        print('Test Passed: Eager and Compile outputs match.')

if __name__ == '__main__':
    test_dropout_eager_compile_divergence()