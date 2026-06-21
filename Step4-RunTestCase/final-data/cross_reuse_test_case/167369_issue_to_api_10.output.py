import torch
import tensorflow as tf
import numpy as np

# Define a custom class similar to Config in the bug report
# This class implements the Keras Regularizer interface to be compatible with the similar API
class CustomRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, factor=0.01):
        self.factor = factor

    def __call__(self, x):
        return self.factor * tf.reduce_sum(x ** 2)

    def get_config(self):
        return {'factor': self.factor}

def forward(x, reg):
    # Using the similar API: tf.keras.regularizers.serialize
    # This is analogous to repr() in the original bug report, converting a custom object to a serializable format
    serialized = tf.keras.regularizers.serialize(reg)
    
    # Using the result in a tensor operation to ensure tracing/compilation handles the conversion
    # len(serialized) works because serialize returns a dictionary
    return x * len(serialized)

# Setup inputs
reg = CustomRegularizer()
x = tf.random.normal((2, 2))

# Compile the function using tf.function (TensorFlow's equivalent of torch.compile)
# This tests if the tracing mechanism can handle the serialization of the custom object
compiled_forward = tf.function(forward)

# Execute the compiled function
result = compiled_forward(x, reg)

# Assertion to verify the test runs successfully and produces the expected shape
assert result.shape == (2, 2), "Output shape mismatch"

print("Test passed. tf.function successfully traced tf.keras.regularizers.serialize on a custom object.")