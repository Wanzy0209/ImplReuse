import torch
import tensorflow as tf
import numpy as np

# Check for device availability (mimicking torch.backends.mps.is_available())
gpus = tf.config.list_physical_devices('GPU')
device_name = '/GPU:0' if gpus else '/CPU:0'

# Wrapper over the tf.experimental.numpy.isfinite operation
# Mimicking MPSSoftshrink from the original bug report
class TFIsFinite(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        # The API under test: tf.experimental.numpy.isfinite
        return tf.experimental.numpy.isfinite(inputs)

# Wrapper over the Sequential layer, using the custom isfinite implementation.
# Mimicking CustomMPSSoftshrinkModel
class CustomTFModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.Sequential([
            TFIsFinite(),
            # Adding a simple activation to mimic the sequence structure
            tf.keras.layers.Activation('linear') 
        ])

    def call(self, x):
        return self.model(x)

# Test execution
with tf.device(device_name):
    model = CustomTFModel()

    # Create input data containing finite, infinite, and NaN values
    # to properly test the isfinite API behavior
    input_data = tf.constant([
        [1.0, 2.5, float('inf')],
        [float('-inf'), float('nan'), 0.0]
    ], dtype=tf.float32)

    # Run the model
    output = model(input_data)

    # Verify results
    # isfinite returns True for finite numbers, False for Inf/NaN
    expected = tf.constant([
        [True, True, False],
        [False, False, True]
    ], dtype=tf.bool)

    # Assert that the output matches the expected boolean mask
    assert tf.reduce_all(tf.equal(output, expected)).numpy(), "Test failed: isfinite output mismatch"

print("Test passed successfully.")