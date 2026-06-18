import torch
import tensorflow as tf
import numpy as np

def test_tf_linear_activation_compile_equivalent():
    """
    Test case adapted from PyTorch Issue 160084.
    This test translates the logic of the PyTorch regression bug to TensorFlow,
    leveraging tf.keras.activations.linear (the similar API) and tf.function 
    (the semantic equivalent of torch.compile).
    """
    
    class RegressionModel(tf.Module):
        def __init__(self, a=0, b=0):
            super().__init__()
            # Using tf.Variable as the equivalent to torch.nn.Parameter
            self.a = tf.Variable(float(a), dtype=tf.float32)
            self.b = tf.Variable(float(b), dtype=tf.float32)
            self.first_batch = True

        # tf.function is the TensorFlow equivalent to torch.compile
        @tf.function
        def __call__(self, x=None):
            if self.first_batch:
                tf.print("Model dtype:", self.a.dtype, ",", self.b.dtype, ". Input dtype:", x.dtype)
                self.first_batch = False
            
            # Preserving the original logic: x * a + b
            # Leveraging the similar API: tf.keras.activations.linear
            # We apply the linear activation to the result of the affine transformation.
            return tf.keras.activations.linear(x * self.a + self.b)

    model = RegressionModel()
    
    # Determine device (GPU or CPU)
    device_name = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
    
    with tf.device(device_name):
        # Create inputs similar to torch.randn(4, 10)
        inputs = tf.random.normal((4, 10), dtype=tf.float32)
        
        # Execute the compiled model
        output = model(inputs)
        
        # Assertions to verify behavior
        # With a=0 and b=0, the output should be all zeros (or very close to it)
        expected_output = inputs * 0.0 + 0.0
        assert output.shape == (4, 10), f"Expected shape (4, 10), got {output.shape}"
        
        # Check if the values match the expected affine transformation
        # Using numpy for assertion clarity
        np.testing.assert_allclose(output.numpy(), expected_output.numpy(), rtol=1e-5)
        
        print("Test passed: Linear activation with tf.function executed successfully.")

if __name__ == "__main__":
    test_tf_linear_activation_compile_equivalent()