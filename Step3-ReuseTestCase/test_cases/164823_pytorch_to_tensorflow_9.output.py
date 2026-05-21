import torch
import tensorflow as tf

def test_model(x):
    # Adapt the logic to use tf.name_scope, which is the identified similar API.
    # This wraps the operations in a namespace, analogous to how torch.compile 
    # wraps the model execution context in the original bug report.
    with tf.name_scope("sparse_operations"):
        # PyTorch: x.to_sparse()
        # TensorFlow: tf.sparse.from_dense
        x_sparse = tf.sparse.from_dense(x)

        # PyTorch: x_sparse * 2
        # TensorFlow: tf.sparse.multiply (supports sparse * dense scalar)
        result_sparse = tf.sparse.multiply(x_sparse, 2.0)

        # PyTorch: result.to_dense()
        # TensorFlow: tf.sparse.to_dense
        result = tf.sparse.to_dense(result_sparse)
        return result

# Input data
x = tf.random.normal((10, 10))

# Run the model
# Note: In TensorFlow, eager execution is the default. 
# tf.name_scope is active here, organizing the operations.
output = test_model(x)

# Verify the output
print("Output shape:", output.shape)
assert output.shape == (10, 10), "Output shape mismatch"
print("Test passed successfully.")