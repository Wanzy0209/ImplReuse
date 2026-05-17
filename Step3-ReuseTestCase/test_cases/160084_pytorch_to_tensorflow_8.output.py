import torch
import tensorflow as tf
import numpy as np

class RegressionModel(tf.keras.Model):
    def __init__(self, a=0, b=0):
        super().__init__()
        # Initialize parameters similar to torch.nn.Parameter
        self.a = tf.Variable(a, dtype=tf.float32, name='a')
        self.b = tf.Variable(b, dtype=tf.float32, name='b')
        self.first_batch = True

    def call(self, x=None):
        if self.first_batch:
            # Mimic the print statement from the original bug report
            print(f"Model dtype: {self.a.dtype}, {self.b.dtype}. Input dtype: {x.dtype}")
            self.first_batch = False
        return x * self.a + self.b

# Adaptation: Use tf.keras.name_scope to wrap the model definition and execution.
# This mirrors the wrapping of model.forward in the original PyTorch code.
with tf.keras.name_scope("regression_scope"):
    model = RegressionModel()
    
    # Create inputs. In PyTorch, inputs were moved to CUDA. 
    # Here we use standard TensorFlow tensors.
    inputs = tf.random.normal((4, 10))
    
    # Run the model
    output = model(inputs)

# Assertions to verify behavior
assert output.shape == (4, 10), "Output shape mismatch"

# Verify that the variables were created within the name scope
# This checks the core functionality of the similar API (name_scope)
assert "regression_scope" in model.a.name, f"Variable 'a' not in scope. Name: {model.a.name}"
assert "regression_scope" in model.b.name, f"Variable 'b' not in scope. Name: {model.b.name}"

print("Test passed successfully.")