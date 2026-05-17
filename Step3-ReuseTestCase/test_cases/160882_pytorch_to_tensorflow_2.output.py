import torch
import tensorflow as tf
import traceback

def f(x: tf.Tensor) -> tf.Tensor:
    """
    Function using the similar API: tf.keras.random.dropout.
    This mirrors the structure of the original PyTorch function which used torch.complex.
    """
    # Using a fixed seed as implied by the context snippet for DTensor dropout
    return tf.keras.random.dropout(x, rate=0.5, seed=42)

# Compile the function using tf.function (analogous to torch.compile)
# In TensorFlow, tf.function traces the graph on the first call.
compiled_f = tf.function(f)

B, F, T = 1, 641, 39

# Input with original shape
x_src = tf.random.normal((B, F, T))
# Input with permuted (mismatched) shape
x_mismatch = tf.random.normal((B, T, F))

print("Testing compiled function with original shape...")
try:
    result_1 = compiled_f(x_src)
    print(f"First call successful. Output shape: {result_1.shape}")
except Exception as e:
    print(f"First call failed: {e}")
    traceback.print_exc()

print("\nTesting compiled function with mismatched shape...")
try:
    # This mirrors the second call in the PyTorch bug report where inputs differ from compilation
    result_2 = compiled_f(x_mismatch)
    print(f"Second call successful. Output shape: {result_2.shape}")
except Exception as e:
    print(f"Second call failed: {e}")
    traceback.print_exc()

# Assertion to verify behavior if no crash occurs
# (In the original bug, this line would not be reached due to AssertionError)
try:
    assert result_2.shape == x_mismatch.shape
    print("\nTest passed: Output shape matches input shape for the mismatched case.")
except AssertionError:
    print("\nTest failed: Output shape mismatch.")