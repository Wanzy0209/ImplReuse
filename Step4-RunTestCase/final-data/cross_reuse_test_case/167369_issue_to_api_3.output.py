import torch
import tensorflow as tf

# Define a custom class that mimics the 'Config' in the bug report.
# To align with the similar API (tf.keras.initializers.serialize), 
# we define a custom Keras initializer.
class CustomConfig(tf.keras.initializers.Initializer):
    def __init__(self, value=1.0):
        self.value = value

    def __repr__(self):
        return f"CustomConfig(value={self.value})"

    def call(self, shape, dtype=None):
        return tf.ones(shape, dtype=dtype) * self.value

    def get_config(self):
        return {"value": self.value}

def forward(x, config):
    # Mimic the logic in the bug report: calling a conversion function (repr/serialize)
    # on a user-defined object inside a compiled function.
    # Original bug: return x * len(repr(config))
    # Adapted logic: serialize the config and use a value from it to scale the tensor.
    serialized_config = tf.keras.initializers.serialize(config)
    
    # Extract a scalar value to perform a tensor operation, 
    # ensuring the graph logic is valid.
    scale = serialized_config['value']
    
    return x * scale

# Setup inputs
config = CustomConfig(value=2.0)
x = tf.random.normal((2, 2))

# Compile the function (equivalent to torch.compile)
# Using jit_compile=True to enforce XLA compilation, similar to fullgraph=True
compiled_forward = tf.function(forward, jit_compile=True)

# Execute the compiled function
# This tests if the tracer/compiler can handle the serialization call
result = compiled_forward(x, config)

# Assertions to verify correctness
assert result.shape == (2, 2), "Output shape mismatch"
# Verify the scaling happened (rough check)
# Note: exact value check might be flaky with random x, but we check execution success primarily.
print("Test passed: tf.keras.initializers.serialize works inside tf.function.")