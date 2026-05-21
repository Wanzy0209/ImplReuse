import torch
import tensorflow as tf
import numpy as np

# Define the function without JIT
def func_nojit(min_val):
    # Using tf.keras.backend.random_uniform
    # We map the 'value' from torch.full to 'minval' here to test dynamic scalar input
    return tf.keras.backend.random_uniform((2,), minval=min_val, maxval=1.0, dtype=tf.float64)

# Define the function with JIT (equivalent to torch.compile)
func_jit = tf.function(func_nojit)

for func in [func_nojit, func_jit]:
    print(f"--- Testing {func.__name__ if hasattr(func, '__name__') else 'Compiled Function'} ---")
    
    # First call with minval = 0.0
    x1 = tf.constant(0.0, dtype=tf.float64)
    res1 = func(x1)
    print(f"Input: {x1.numpy()}, Output: {res1.numpy()}")
    
    # Second call with minval = 0.5
    x2 = tf.constant(0.5, dtype=tf.float64)
    res2 = func(x2)
    print(f"Input: {x2.numpy()}, Output: {res2.numpy()}")
    
    # Assertion to check if the value was updated
    # If the bug exists (caching), res2 might contain values < 0.5
    # We check if all values are >= 0.5
    try:
        assert np.all(res2.numpy() >= 0.5), "Bug detected: minval argument was cached!"
        print("Assertion Passed: Values are correctly updated.")
    except AssertionError as e:
        print(e)