import tensorflow as tf
import numpy as np

# Fix: Enable eager execution to allow .numpy() method on Tensors.
# The error 'Tensor' object has no attribute 'numpy' typically occurs
# when TensorFlow is running in graph mode (TF 1.x default or disabled eager).
tf.compat.v1.enable_eager_execution()

def test_tf_shape_n_issue_164157():
    """
    Test case for tf.shape_n based on the tensor shapes and dtypes 
    found in Issue 164157 (PyTorch std divergence).
    
    The original issue involved float16 and int64 tensors with specific shapes.
    This test verifies that tf.shape_n correctly handles these tensor configurations.
    """
    # Replicate tensor creation from the PyTorch bug report
    # t6, t7, t8: size=(256, 88, 1), dtype=float16
    t6 = tf.random.uniform((256, 88, 1), dtype=tf.float16)
    t7 = tf.random.uniform((256, 88, 1), dtype=tf.float16)
    t8 = tf.random.uniform((256, 88, 1), dtype=tf.float16)

    # t0: size=(47,), dtype=int64
    t0 = tf.random.uniform((47,), maxval=1000, dtype=tf.int64)

    # The PyTorch code used a list of tensors [t6, t6, t7, t8] for concatenation.
    # We use a similar list structure for tf.shape_n.
    input_tensors = [t6, t6, t7, t8]

    # Call the similar API: tf.shape_n
    # We use out_type=tf.int64 to match the int64 usage in the original issue
    shapes = tf.shape_n(input_tensors, out_type=tf.int64)

    # Assertions
    # 1. Check that we get the correct number of shape outputs
    assert len(shapes) == 4, f"Expected 4 shapes, got {len(shapes)}"

    # 2. Verify the shape content matches the input tensor dimensions
    expected_shape = np.array([256, 88, 1], dtype=np.int64)
    for i, shape_tensor in enumerate(shapes):
        actual_shape = shape_tensor.numpy()
        assert np.array_equal(actual_shape, expected_shape), \
            f"Shape mismatch at index {i}. Expected {expected_shape}, got {actual_shape}"

    # 3. Verify handling of the int64 tensor (t0)
    int_shapes = tf.shape_n([t0], out_type=tf.int64)
    assert np.array_equal(int_shapes[0].numpy(), [47]), \
        f"Shape mismatch for int64 tensor. Expected [47], got {int_shapes[0].numpy()}"

    print("Test Passed: tf.shape_n correctly handles float16 and int64 tensor shapes.")

if __name__ == "__main__":
    test_tf_shape_n_issue_164157()