import tensorflow as tf

# Enable eager execution (The API under test)
# Note: This must be called at the very beginning of the program.
tf.compat.v1.enable_eager_execution()

# Verify that eager execution is enabled
assert tf.executing_eagerly(), "TensorFlow should be in eager execution mode"

# Define a simple model to mimic the FluxPipeline structure
class SimpleModel(tf.Module):
    def __init__(self):
        self.dense = tf.keras.layers.Dense(10)

    def __call__(self, x):
        return self.dense(x)

# Initialize the model
model = SimpleModel()

# Run inference (mimicking the pipe() call in the original test case)
# In eager mode, operations execute immediately and return concrete values,
# rather than building a graph to be executed later.
input_data = tf.constant([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]], dtype=tf.float32)
output = model(input_data)

# Verify the output is a concrete tensor (eager behavior)
# We can access .numpy() immediately because the operation has already executed.
print(f"Output: {output.numpy()}")
assert output.shape == (1, 10)