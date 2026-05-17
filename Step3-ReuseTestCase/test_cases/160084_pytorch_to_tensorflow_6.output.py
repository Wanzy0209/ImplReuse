import torch
import tensorflow as tf
import numpy as np

class RegressionModel(tf.Module):
    def __init__(self, a=0, b=0):
        super().__init__()
        # Define parameters similar to torch.nn.Parameter
        self.a = tf.Variable(float(a), dtype=tf.float32)
        self.b = tf.Variable(float(b), dtype=tf.float32)
        # Use tf.Variable for state to ensure it works in graph mode (tf.function)
        self.first_batch = tf.Variable(True, dtype=tf.bool)

    # Mimic torch.compile behavior using tf.function (graph compilation)
    @tf.function
    def __call__(self, x=None):
        # Use the target API: tf.compat.v1.name_scope to structure the graph
        with tf.compat.v1.name_scope("regression_model"):
            # Autograph will handle the conditional logic based on the Variable
            if self.first_batch:
                # tf.print is used here for graph-mode compatibility instead of standard print
                tf.print("Model dtype:", self.a.dtype, ",", self.b.dtype, ". Input dtype:", x.dtype)
                self.first_batch.assign(False)
            return x * self.a + self.b

if __name__ == "__main__":
    # Initialize model
    model = RegressionModel()
    
    # Check for CUDA availability and set device
    gpus = tf.config.list_physical_devices('GPU')
    device_name = '/GPU:0' if gpus else '/CPU:0'
    
    with tf.device(device_name):
        # Create inputs similar to torch.randn
        inputs = tf.random.normal((4, 10))
        
        # Run the model
        output = model(inputs)
        
        # Basic assertion to verify execution
        assert output.shape == (4, 10), "Output shape mismatch"
        
        print(f"Test executed successfully on {device_name}.")