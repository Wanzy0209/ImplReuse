import torch
import numpy as np
import sys

# Attempt to import TensorFlow and handle environment errors gracefully
try:
    import tensorflow as tf
except ImportError as e:
    # Check for the specific GLIBCXX error mentioned in the traceback
    if "GLIBCXX" in str(e):
        print("Skipping test: TensorFlow import failed due to environment issue (GLIBCXX version mismatch).")
        print(f"Details: {e}")
        sys.exit(0)
    else:
        # If it's a different import error, re-raise it
        raise

# Ensure V1 behavior is active to match the API context
tf.compat.v1.disable_eager_execution()

# Dimensions from the original bug report
m = 20120
k = 1536
n = 512

def test_tf_compat_v1_addmm():
    """
    Test case adapted from PyTorch Inductor/Triton issue (ID: 161618).
    Replicates the torch.addmm logic using tf.compat.v1 API patterns.
    """
    # Define placeholders for inputs (mimicking torch.randn(...).cuda())
    # In TF v1, we define the graph structure first.
    a = tf.compat.v1.placeholder(tf.float32, shape=[m, n], name='a')
    mat1 = tf.compat.v1.placeholder(tf.float32, shape=[m, k], name='mat1')
    mat2 = tf.compat.v1.placeholder(tf.float32, shape=[k, n], name='mat2')

    # Reproduce torch.addmm(a, mat1, mat2) logic
    # addmm computes beta * input + alpha * (mat1 @ mat2). Default alpha=1, beta=1.
    # TensorFlow equivalent: tf.add(a, tf.matmul(mat1, mat2))
    matmul_result = tf.matmul(mat1, mat2)
    result = tf.add(a, matmul_result, name='addmm_result')

    # Initialize variables and session
    init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init)

        # Generate random data to feed the graph (mimicking torch.randn)
        feed_dict = {
            a: np.random.randn(m, n).astype(np.float32),
            mat1: np.random.randn(m, k).astype(np.float32),
            mat2: np.random.randn(k, n).astype(np.float32)
        }

        # Execute the operation
        output = sess.run(result, feed_dict=feed_dict)

        # Assertions to verify the operation executed correctly
        assert output.shape == (m, n), f"Output shape mismatch: expected {(m, n)}, got {output.shape}"
        
        # Verify numerical correctness against numpy
        expected_a = feed_dict[a]
        expected_matmul = np.dot(feed_dict[mat1], feed_dict[mat2])
        expected_output = expected_a + expected_matmul
        
        np.testing.assert_allclose(output, expected_output, rtol=1e-5)

    print("Test passed: tf.compat.v1 addmm execution successful.")

if __name__ == "__main__":
    test_tf_compat_v1_addmm()