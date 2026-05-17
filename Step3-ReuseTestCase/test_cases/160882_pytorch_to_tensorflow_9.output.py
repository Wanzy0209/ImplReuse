import torch
import tensorflow as tf
import tensorflow.experimental.numpy as tnp

def f(arr1: tf.Tensor, arr2: tf.Tensor):
    # The target API: tf.experimental.numpy.broadcast_arrays
    # This function broadcasts input arrays to a common shape.
    return tnp.broadcast_arrays(arr1, arr2)

# Wrap the function with tf.function to simulate the compilation/graph capture behavior
# similar to torch.compile
compiled_f = tf.function(f)

B, F, T = 1, 641, 39

# Create source tensors with shape (B, F, T)
src1 = tf.random.normal((B, F, T))
src2 = tf.random.normal((B, F, T))

# Create mismatched tensors by permuting dimensions to (B, T, F)
mismatch1 = tf.transpose(src1, [0, 2, 1])
mismatch2 = tf.transpose(src2, [0, 2, 1])

print("Testing with source shapes:", src1.shape, src2.shape)
# First call: Trace/Compile with shape (1, 641, 39)
res_src = compiled_f(src1, src2)
print("Result shapes:", [r.shape for r in res_src])

print("\nTesting with mismatched shapes:", mismatch1.shape, mismatch2.shape)
# Second call: Execute with shape (1, 39, 641)
# In the original PyTorch bug, this change in input shapes after compilation
# caused a crash. This test verifies if the TensorFlow API handles the dynamic
# shape change gracefully within a compiled context.
try:
    res_mismatch = compiled_f(mismatch1, mismatch2)
    print("Result shapes:", [r.shape for r in res_mismatch])
    print("Test passed: The API handled the input shape change correctly.")
except Exception as e:
    print(f"Test failed with error: {e}")