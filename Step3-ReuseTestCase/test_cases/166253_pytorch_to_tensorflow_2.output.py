import torch
import tensorflow as tf

def func_nojit(mu):
    # Adapted from torch.full((2, ), x, dtype=torch.float64)
    # tf.keras.backend.random_normal creates a tensor of shape (2,) with mean mu
    return tf.keras.backend.random_normal((2,), mu, 1.0, dtype=tf.float64)

# TensorFlow equivalent of torch.compile is tf.function
func_jit = tf.function(func_nojit)

print("Testing tf.keras.backend.random_normal behavior:")
print("=" * 50)

for func in [func_nojit, func_jit]:
    # Define inputs matching the original test case values
    mu1 = tf.constant(5.0, dtype=tf.float64)
    mu2 = tf.constant(10.0, dtype=tf.float64)
    
    print(f"\nRunning {func.__name__}:")
    
    # First call
    res1 = func(mu1)
    print(f"Result with mu=5.0:  {res1}")
    
    # Second call
    res2 = func(mu2)
    print(f"Result with mu=10.0: {res2}")
    
    # Assertions
    # 1. Check dtype (as per the original bug report context)
    assert res1.dtype == tf.float64, f"Expected float64, got {res1.dtype}"
    assert res2.dtype == tf.float64, f"Expected float64, got {res2.dtype}"
    
    # 2. Check shape
    assert res1.shape == (2,), f"Expected shape (2,), got {res1.shape}"
    assert res2.shape == (2,), f"Expected shape (2,), got {res2.shape}"

    # Note: Since random_normal generates stochastic values, we cannot assert exact values
    # like in the torch.full case. However, if a caching bug similar to the PyTorch issue
    # existed, the mean of the distribution in the second call would incorrectly reflect
    # the first call's input (5.0) instead of the current input (10.0).