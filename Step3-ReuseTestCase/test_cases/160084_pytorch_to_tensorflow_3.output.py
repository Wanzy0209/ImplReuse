import torch
import tensorflow as tf

# Enable eager execution using the similar API provided.
# In TensorFlow 1.x compatibility mode, this enables the imperative execution
# style similar to PyTorch's default, contrasting with graph compilation.
tf.compat.v1.enable_eager_execution()

class RegressionModel(tf.Module):
    def __init__(self, a=0, b=0):
        super().__init__()
        # Define parameters as TensorFlow Variables
        self.a = tf.Variable(float(a), dtype=tf.float32)
        self.b = tf.Variable(float(b), dtype=tf.float32)
        self.first_batch = True

    def __call__(self, x=None):
        # Preserve the logic from the original bug report
        if self.first_batch:
            print(f"Model dtype: {self.a.dtype}, {self.b.dtype}. Input dtype: {x.dtype}")
            self.first_batch = False
        # Preserve the arithmetic operation
        return x * self.a + self.b

def test_eager_execution_model():
    # Initialize the model
    model = RegressionModel()
    
    # Create random inputs (TensorFlow equivalent of torch.randn)
    # Note: In eager execution, we don't explicitly move to device in the same way,
    # operations execute immediately on the available device.
    inputs = tf.random.normal((4, 10), dtype=tf.float32)
    
    # Execute the model forward pass
    output = model(inputs)
    
    # Assertions to verify behavior
    assert output.shape == (4, 10), "Output shape mismatch"
    assert output.dtype == tf.float32, "Output dtype mismatch"
    
    # Verify that the internal state (first_batch flag) was updated
    assert model.first_batch == False, "Internal state flag did not update correctly"
    
    print("Test passed: Eager execution handled the model logic correctly.")

if __name__ == "__main__":
    test_eager_execution_model()