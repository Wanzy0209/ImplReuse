import tensorflow as tf
import numpy as np

def test_keras_backend_ndim():
    """
    Test case for tf.keras.backend.ndim, adapted from the logic of 
    Issue 162630 (Intrusive Caching DLPack).
    
    The original issue highlights the overhead of frequent metadata access/conversion 
    (DLPack) and proposes caching to optimize repeated calls. 
    This test verifies the correctness and consistency of metadata access 
    (specifically dimensionality/ndim) for the similar API, ensuring that 
    repeated calls return the expected result without error, mirroring the 
    'frequent tensor exchanges' scenario described in the bug report.
    """
    
    # Scenario 1: Symbolic Tensor (Placeholder)
    # Mirrors the need to access metadata on tensors that might be used multiple times
    input_shape = (2, 4, 5)
    input_tensor = tf.keras.backend.placeholder(shape=input_shape)
    
    # First access
    rank = tf.keras.backend.ndim(input_tensor)
    assert rank == 3, f"Expected rank 3 for shape {input_shape}, got {rank}"
    
    # Scenario 2: Variable (Keras Variable)
    # Mirrors the 'model weights' scenario mentioned in the bug description
    val = np.array([[1, 2], [3, 4]])
    kvar = tf.keras.backend.variable(value=val)
    
    # First access
    rank_var = tf.keras.backend.ndim(kvar)
    assert rank_var == 2, f"Expected rank 2 for variable, got {rank_var}"
    
    # Scenario 3: Frequent Access Simulation
    # The bug report mentions "frequent tensor exchanges... can accumulate this overhead."
    # We verify that the API remains consistent under repeated calls.
    # In the context of the original issue, this is where the caching mechanism would be critical.
    for _ in range(100):
        assert tf.keras.backend.ndim(input_tensor) == 3
        assert tf.keras.backend.ndim(kvar) == 2
        
    # Scenario 4: Constant Tensor
    const_tensor = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert tf.keras.backend.ndim(const_tensor) == 3

if __name__ == "__main__":
    test_keras_backend_ndim()
    print("Test passed.")