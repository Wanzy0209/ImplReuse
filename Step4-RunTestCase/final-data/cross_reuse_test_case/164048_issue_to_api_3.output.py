import tensorflow as tf
import numpy as np

def test_large_tensor_lu_solve():
    """
    Test case for tf.linalg.lu_solve adapted from PyTorch Issue 164048.
    
    The original issue involved an 'invalid configuration argument' error when 
    performing boolean indexing on a large tensor with dimensions (4, 87, 1056, 736) on CUDA.
    
    This test adapts the 'large tensor' stress logic to the tf.linalg.lu_solve API.
    We map the problematic dimensions to the inputs of lu_solve to check if 
    similar large-scale operations trigger configuration errors.
    """
    
    # Dimensions derived from the original bug report: (4, 87, 1056, 736)
    # Mapping:
    # batch_size = 4
    # matrix_size (M) = 1056
    # rhs_cols (K) = 736
    
    batch_size = 4
    matrix_size = 1056
    rhs_cols = 736

    # Create large tensors mimicking the scale of the original bug
    # lower_upper shape: [batch_size, matrix_size, matrix_size]
    lower_upper = tf.random.normal((batch_size, matrix_size, matrix_size), dtype=tf.float32)
    
    # perm shape: [batch_size, matrix_size]
    # We generate a valid permutation for each batch element
    perm = tf.stack([tf.random.shuffle(tf.range(matrix_size, dtype=tf.int32)) for _ in range(batch_size)])
    
    # rhs shape: [batch_size, matrix_size, rhs_cols]
    rhs = tf.random.normal((batch_size, matrix_size, rhs_cols), dtype=tf.float32)

    # Attempt to execute the operation
    # The original bug was specific to CUDA, so we try to use GPU if available
    
    # Fix: Handle different TensorFlow versions for GPU detection
    has_gpu = False
    try:
        gpus = tf.config.list_physical_devices('GPU')
        has_gpu = len(gpus) > 0
    except AttributeError:
        # Fallback for older TensorFlow versions (e.g., 1.x or early 2.x)
        try:
            has_gpu = tf.test.is_gpu_available()
        except:
            # If detection fails completely, default to CPU
            has_gpu = False

    device_name = "/GPU:0" if has_gpu else "/CPU:0"
    
    try:
        with tf.device(device_name):
            # Perform the solve operation
            result = tf.linalg.lu_solve(lower_upper, perm, rhs)
            
            # Verify the output shape matches expectations
            assert result.shape == (batch_size, matrix_size, rhs_cols), \
                f"Shape mismatch. Expected {(batch_size, matrix_size, rhs_cols)}, got {result.shape}"
            
        print(f"Test passed: tf.linalg.lu_solve handled large tensors on {device_name} successfully.")

    except tf.errors.InvalidArgumentError as e:
        # Catching the specific error type mentioned in the PyTorch bug title
        print(f"Test failed with InvalidArgumentError (similar to original bug): {e}")
        raise
    except Exception as e:
        print(f"Test failed with unexpected error: {e}")
        raise

if __name__ == "__main__":
    test_large_tensor_lu_solve()