import torch
import tensorflow as tf

# Adapted test case for tf.keras.initializers.HeNormal
# Original Bug Context: Eager vs Compile divergence when handling tensors with 
# zero-sized dimensions (specifically shape (20, 0)).
# Adaptation Logic: Verify that the HeNormal initializer handles zero-sized 
# dimensions consistently in both Eager mode and Graph (tf.function) mode.

def test_initializer(shape):
    # Initialize HeNormal with the seed from the original bug report
    initializer = tf.keras.initializers.HeNormal(seed=1014698)
    return initializer(shape=shape)

# The problematic shape identified in the PyTorch bug report
target_shape = (20, 0)

print(f"Testing tf.keras.initializers.HeNormal with shape: {target_shape}")

# 1. Test Eager Execution
try:
    result_eager = test_initializer(target_shape)
    assert result_eager.shape == target_shape, f"Eager shape mismatch: expected {target_shape}, got {result_eager.shape}"
    assert result_eager.dtype == tf.float32, f"Eager dtype mismatch: expected float32, got {result_eager.dtype}"
    print(f" Eager success. Shape: {result_eager.shape}, Dtype: {result_eager.dtype}")
except Exception as e:
    print(f" Eager failure: {e}")
    raise

# 2. Test Compiled/Graph Execution (tf.function)
# This corresponds to the 'torch.compile' aspect of the original bug
try:
    compiled_test_initializer = tf.function(test_initializer)
    result_compiled = compiled_test_initializer(target_shape)
    assert result_compiled.shape == target_shape, f"Compiled shape mismatch: expected {target_shape}, got {result_compiled.shape}"
    assert result_compiled.dtype == tf.float32, f"Compiled dtype mismatch: expected float32, got {result_compiled.dtype}"
    print(f" Compile (tf.function) success. Shape: {result_compiled.shape}, Dtype: {result_compiled.dtype}")
except Exception as e:
    print(f" Compile failure: {e}")
    raise

print(" All tests passed. No divergence detected.")