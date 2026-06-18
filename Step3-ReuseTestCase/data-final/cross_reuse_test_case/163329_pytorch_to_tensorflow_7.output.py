import torch
import tensorflow as tf
import logging

# 1. Setup logging to monitor recompilation (retracing)
# Equivalent to torch._logging.set_logs(recompiles=True)
logging.getLogger("tensorflow").setLevel(logging.INFO)

# 2. Define a mock pipeline structure similar to the original issue
class MockTransformer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(10)

    # Using the similar API: tf.keras.backend.name_scope
    # This represents the "region" being compiled
    def call(self, inputs):
        with tf.keras.backend.name_scope("transformer_region"):
            return self.dense(inputs)

class MockPipeline(tf.Module):
    def __init__(self):
        self.transformer = MockTransformer()

    # Equivalent to torch.compile
    @tf.function
    def __call__(self, inputs):
        return self.transformer(inputs)

# 3. Initialize pipeline
pipe = MockPipeline()

# 4. Prepare inputs
# Mimicking the prompt input as a tensor
inputs = tf.random.normal((1, 10))

# 5. Run inference
# First call triggers compilation (tracing)
print(">>> First call (Compilation/Tracing)")
output = pipe(inputs)

# Second call should use the compiled graph (no recompilation)
print(">>> Second call (Cached execution)")
output = pipe(inputs)

# Verify output shape to ensure correctness
assert output.shape == (1, 10)