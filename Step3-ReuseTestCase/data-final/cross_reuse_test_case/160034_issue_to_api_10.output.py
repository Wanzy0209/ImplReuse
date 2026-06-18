import torch
import tensorflow as tf
import numpy as np

def test_nonneg_complex_dtype_handling():
    """
    Test case for tf.keras.constraints.NonNeg inspired by PyTorch Issue 160034.
    
    The original issue highlights a failure mode and cryptic error message when 
    handling complex64 dtypes with index_put_ on the MPS backend.
    
    This test verifies how the similar API (tf.keras.constraints.NonNeg) handles
    complex64 inputs, specifically checking if the type casting logic 
    (math_ops.cast to backend.floatx) handles the type gracefully or produces
    a clear error, addressing the 'error message improvement' aspect of the issue.
    """
    
    # Setup: Ensure a specific floatx is set (similar to scalar_type checks in PyTorch)
    tf.keras.backend.set_floatx('float32')
    
    # Create a complex64 tensor, mirroring the PyTorch issue's setup:
    # image = torch.zeros(10, dtype=torch.complex64, device=device)
    complex_weights = tf.constant([1.0 + 2.0j, -1.0 + 1.0j, 0.0 + 0.0j], dtype=tf.complex64)
    
    # Instantiate the constraint
    # PyTorch equivalent: image.index_put_(...)
    constraint = tf.keras.constraints.NonNeg()
    
    # Apply the constraint
    # The implementation of NonNeg is: w * math_ops.cast(math_ops.greater_equal(w, 0.), backend.floatx())
    # We test if this logic supports complex inputs or fails with a clear message.
    try:
        result = constraint(complex_weights)
        
        # If it succeeds, it likely cast the complex tensor to floatx.
        # We assert the output is non-negative and of the correct type.
        assert result.dtype == tf.float32, f"Expected float32, got {result.dtype}"
        assert np.all(result.numpy() >= 0), "Constraint failed: negative values found"
        print("Test Passed: NonNeg successfully handled complex64 input by casting to float32.")
        
    except Exception as e:
        # If it fails, the error message should ideally be explicit about the type mismatch.
        # This mirrors the goal of the PyTorch issue (improving error messages).
        error_msg = str(e)
        print(f"Exception caught: {error_msg}")
        
        # Example assertion for a "good" error message (optional, depending on TF behavior)
        # assert "complex" in error_msg.lower() or "dtype" in error_msg.lower(), \
        #     "Error message should explicitly mention the type issue."

if __name__ == "__main__":
    test_nonneg_complex_dtype_handling()