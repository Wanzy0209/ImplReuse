import torch
import tensorflow as tf
import numpy as np
import sys

# Ensure reproducibility
tf.random.set_seed(42)

def run_test(seed_val):
    # Adaptation: Use tf.keras.backend.random_normal to generate the tensors
    # that would have been the result of torch.addmm and other inputs.
    # Original: t3 = torch.addmm(t0, t1, t2) -> size=(5, 4), dtype=bfloat16
    # We simulate the output of the linear operation directly with random_normal
    # to test the numerical stability of the subsequent graph in different modes.
    t3 = tf.keras.backend.random_normal(
        shape=(5, 4), 
        mean=0.0, 
        stddev=1.0, 
        dtype=tf.bfloat16, 
        seed=seed_val
    )

    # t4 = t3.norm() -> size=(), dtype=bfloat16
    t4 = tf.norm(t3)

    # t5 = arg3 -> size=(3, 4, 5, 2), dtype=float32
    t5 = tf.keras.backend.random_normal(
        shape=(3, 4, 5, 2), 
        mean=0.0, 
        stddev=1.0, 
        dtype=tf.float32, 
        seed=seed_val + 1
    )
    
    # t6 = t5.var(dim=0) -> size=(4, 5, 2), dtype=float32
    t6 = tf.math.reduce_variance(t5, axis=0)
    
    # t7 = t6.var() -> size=(), dtype=float32
    t7 = tf.math.reduce_variance(t6)

    # t8 = arg4 -> size=(), dtype=float32
    t8 = tf.keras.backend.random_normal(
        shape=[], 
        mean=0.0, 
        stddev=1.0, 
        dtype=tf.float32, 
        seed=seed_val + 2
    )
    
    # t9 = torch.nn.functional.relu(t8)
    t9 = tf.nn.relu(t8)

    # t10 = t7 + t4 + t9
    # Cast t4 to float32 to match PyTorch's implicit promotion logic in the original bug
    t4_float = tf.cast(t4, tf.float32)
    t10 = t7 + t4_float + t9

    # t11 = torch.pow(torch.pow(t4, t7), t10)
    # Note: t4 is cast to float32 here for the power operation
    t11 = tf.pow(tf.pow(t4_float, t7), t10)
    
    return t11

if __name__ == '__main__':
    # 1. Eager Execution
    print("Running Eager Execution...")
    out_eager = run_test(100)
    print('Eager Success! ')
    
    # 2. Compiled Execution (using tf.function with XLA)
    # This mimics torch.compile(fullgraph=True)
    print("Running Compiled Execution...")
    try:
        compiled_foo = tf.function(run_test, jit_compile=True)
        out_compiled = compiled_foo(100)
        print('Compile Success! ')
    except Exception as e:
        print(f"Compile failed (XLA might not be available or other issue): {e}")
        sys.exit(1)

    # 3. Compare outputs
    # Extract scalar values
    val_eager = out_eager.numpy()
    val_compiled = out_compiled.numpy()
    
    diff = np.abs(val_eager - val_compiled)
    # Avoid division by zero
    denom = np.abs(val_eager) + 1e-12
    rel_diff = (diff / denom) * 100
    
    print(f'Relative diff: {rel_diff:.6f}%')
    print(f'Eager value: {val_eager}')
    print(f'Compiled value: {val_compiled}')
    
    # Check for significant divergence (threshold from original bug: 5%)
    if rel_diff > 5:
        print(f' Outputs differ significantly!')
        print('Absolute diff:', diff)
        sys.exit(1)
    else:
        print(' Outputs match within tolerance.')