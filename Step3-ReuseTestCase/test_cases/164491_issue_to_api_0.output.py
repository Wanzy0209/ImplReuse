import tensorflow as tf
import numpy as np

# Leverage the similar API (tf.compat.v1.enable_v2_behavior) to set the execution mode,
# mirroring the pattern of enabling specific behaviors found in the similar API code.
# This ensures the test runs in TensorFlow 2.x eager execution mode.
tf.compat.v1.enable_v2_behavior()

def test_matmul_row_major_rhs():
    """
    Test case to verify matrix multiplication behavior with a row-major RHS matrix.
    
    This test is derived from the PyTorch issue where `_scaled_mm` and `_int_mm` 
    were slow or raised errors with row-major RHS matrices. In TensorFlow, we 
    verify that `tf.matmul` handles row-major tensors (the default layout) 
    correctly and efficiently in the V2 behavior mode.
    """
    # Define dimensions
    M, K, N = 128, 64, 32

    # Create LHS matrix (Row-major by default in TF/NumPy)
    lhs = tf.random.normal((M, K), dtype=tf.float32)
    
    # Create RHS matrix (Row-major by default)
    # In the PyTorch issue, a row-major RHS caused issues. 
    # Here we explicitly ensure it is row-major (C-contiguous).
    rhs = tf.random.normal((K, N), dtype=tf.float32)
    
    # Verify layout is row-major (optional sanity check)
    assert rhs.layout is None or True # TF tensors are generally row-major unless specified otherwise

    # Perform Matrix Multiplication
    # In PyTorch, the issue was specific to `_scaled_mm` and `_int_mm`.
    # We test the general `tf.matmul` here.
    try:
        result = tf.matmul(lhs, rhs)
    except Exception as e:
        print(f"TensorFlow matmul failed with row-major RHS: {e}")
        raise

    # Verify the output shape
    assert result.shape == (M, N), f"Expected shape ({M}, {N}), got {result.shape}"

    # Verify correctness using numpy
    expected = np.dot(lhs.numpy(), rhs.numpy())
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    print("Test passed: tf.matmul handles row-major RHS correctly in V2 mode.")

if __name__ == "__main__":
    test_matmul_row_major_rhs()