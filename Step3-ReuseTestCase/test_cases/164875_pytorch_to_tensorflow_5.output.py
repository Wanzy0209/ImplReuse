import torch
import tensorflow as tf
import numpy as np

# Adapt the PyTorch test case logic to TensorFlow.
# The original bug involves handling tensors with a zero-sized dimension (20, 0)
# during eager execution vs. compiled execution (torch.compile).
# We test if tf.keras.initializers.HeUniform handles this edge case correctly
# in both eager and graph (tf.function) modes.

# Set seed for reproducibility, matching the original report's seed
tf.random.set_seed(1014698)

# The problematic shape identified in the PyTorch bug report
# Shape (20, 0) implies a batch size of 20 but 0 features (or vice versa)
target_shape = (20, 0)

def test_he_uniform_edge_case():
    print(f"Testing tf.keras.initializers.HeUniform with shape: {target_shape}")
    
    # Initialize the API under test
    # HeUniform is a VarianceScaling initializer. 
    # Logic: limit = sqrt(6 / fan_in). If fan_in is 0 (due to shape 20,0), 
    # this might cause a division by zero or shape mismatch errors.
    initializer = tf.keras.initializers.HeUniform(seed=1014698)

    # 1. Test in Eager Mode
    print("\n--- Testing Eager Mode ---")
    try:
        # Attempt to generate a tensor with the zero-sized dimension
        result_eager = initializer(target_shape)
        print(f" Eager success. Shape: {result_eager.shape}, Dtype: {result_eager.dtype}")
    except Exception as e:
        print(f" Eager failed: {type(e).__name__}: {e}")
        return

    # 2. Test in Graph/Compiled Mode (tf.function)
    # This mirrors the torch.compile(fullgraph=True) behavior in the original bug
    print("\n--- Testing Graph Mode (tf.function) ---")
    
    @tf.function
    def compiled_init(init, shape):
        return init(shape)

    try:
        result_compiled = compiled_init(initializer, target_shape)
        print(f" Compile success. Shape: {result_compiled.shape}, Dtype: {result_compiled.dtype}")
    except Exception as e:
        print(f" Compile failed: {type(e).__name__}: {e}")
        return

    # 3. Verify Consistency
    # Check for divergence between eager and compiled modes
    print("\n--- Verifying Consistency ---")
    if result_eager.shape != result_compiled.shape:
        print(f" Divergence detected: Eager shape {result_eager.shape} != Compiled shape {result_compiled.shape}")
    else:
        print(f" Consistency check passed: Shapes match {result_eager.shape}.")
        
        # Note: We cannot compare values (np.allclose) because the tensor is empty (size 0).
        # The critical check is that both modes handle the zero-dimension without crashing
        # and produce the same shape metadata.

if __name__ == "__main__":
    test_he_uniform_edge_case()