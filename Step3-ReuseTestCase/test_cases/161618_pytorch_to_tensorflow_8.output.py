import torch
import tensorflow as tf

# Adapted from PyTorch test case to TensorFlow
# Original Bug: Inductor fails on triton main branch during torch.addmm compilation
# Target API: tf.keras.name_scope (Context manager for naming operations)

def test_tf_name_scope_addmm():
    # Dimensions from the original bug report
    m = 20120
    k = 1536
    n = 512

    # Create tensors (equivalent to torch.randn(...).cuda())
    # Using float32 as in the original script
    a = tf.random.normal((m, n), dtype=tf.float32)
    mat1 = tf.random.normal((m, k), dtype=tf.float32)
    mat2 = tf.random.normal((k, n), dtype=tf.float32)

    # Define the function equivalent to torch.addmm(a, mat1, mat2)
    # torch.addmm computes: mat1 * mat2 + a
    f = lambda a, mat1, mat2: a + tf.matmul(mat1, mat2)

    # Use the similar API: tf.keras.name_scope
    # This replaces the 'inductor_config.patch' context manager from the original script.
    # While name_scope doesn't configure compilation backends like Inductor,
    # it serves as the structural equivalent for wrapping execution context.
    with tf.keras.name_scope("addmm_operation"):
        result = f(a, mat1, mat2)

    # Verify the result shape matches the expected output
    assert result.shape == (m, n), f"Expected shape ({m}, {n}), but got {result.shape}"
    
    # Verify the operation executed successfully (no crash)
    print("Test passed: tf.keras.name_scope handled the addmm operation successfully.")

if __name__ == "__main__":
    test_tf_name_scope_addmm()