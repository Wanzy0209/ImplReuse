import torch
import tensorflow as tf
import numpy as np

class SparsePlaceholderTest:
    """
    Test case for tf.compat.v1.sparse_placeholder adapted from the 
    torch.isin scalar bug report pattern.
    
    The original bug (Issue 164924) involved torch.isin failing when 
    'test_elements' was a scalar (0-d tensor) during compilation.
    
    This test adapts that logic to TensorFlow by:
    1. Using a class-based structure similar to the PyTorch Module.
    2. Testing the API (sparse_placeholder) in graph mode (analogous to torch.compile).
    3. Specifically testing a scalar-shaped input to mimic the bug's trigger condition.
    """
    def __init__(self):
        super().__init__()
        np.random.seed(777)
        
        # Setup 1: Standard sparse tensor (analogous to self.x in bug report)
        self.indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
        self.values = np.array([1.0, 2.0], dtype=np.float32)
        self.shape = np.array([3, 4], dtype=np.int64)

        # Setup 2: Scalar sparse tensor (analogous to self.y in bug report)
        # A scalar sparse tensor has shape [] and indices [[]]
        self.scalar_indices = np.array([[]], dtype=np.int64)
        self.scalar_values = np.array([5.0], dtype=np.float32)
        self.scalar_shape = np.array([], dtype=np.int64)

        # Define the graph using the similar API
        # tf.compat.v1.sparse_placeholder is the graph-mode input mechanism
        self.sparse_ph = tf.compat.v1.sparse_placeholder(dtype=tf.float32, shape=None)
        
        # Define a computation to verify execution (reduce_sum)
        self.result = tf.sparse.reduce_sum(self.sparse_ph)

    def run(self, indices, values, shape):
        # Execute the graph (analogous to compiled_model.forward())
        with tf.compat.v1.Session() as sess:
            output = sess.run(self.result, feed_dict={
                self.sparse_ph: (indices, values, shape)
            })
            return output

if __name__ == "__main__":
    # Disable eager execution to simulate the 'compile' environment
    tf.compat.v1.disable_eager_execution()
    
    model = SparsePlaceholderTest()
    
    # Test 1: Standard tensor execution
    try:
        res_standard = model.run(model.indices, model.values, model.shape)
        print(f"Standard Tensor Result: {res_standard}")
        assert res_standard == 3.0, "Standard tensor test failed"
    except Exception as e:
        print(f"Standard Tensor Error: {e}")

    # Test 2: Scalar tensor execution (Mimicking the bug report's scalar input)
    try:
        res_scalar = model.run(model.scalar_indices, model.scalar_values, model.scalar_shape)
        print(f"Scalar Tensor Result: {res_scalar}")
        assert res_scalar == 5.0, "Scalar tensor test failed"
    except Exception as e:
        print(f"Scalar Tensor Error: {e}")

    print("Test completed.")