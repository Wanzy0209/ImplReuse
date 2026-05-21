import torch
import tensorflow as tf
import numpy as np

# Check for hardware availability (analogous to torch.backends.mps.is_available())
gpus = tf.config.list_physical_devices('GPU')
assert len(gpus) > 0, "GPU is not available"

# Wrapper over the target API (analogous to MPSSoftshrink)
class ResizeBilinearLayer(tf.keras.layers.Layer):
    def __init__(self, size=(64, 64), **kwargs):
        super().__init__(**kwargs)
        self.size = size

    def call(self, inputs):
        # Using the similar API: tf.compat.v1.image.resize_bilinear
        return tf.compat.v1.image.resize_bilinear(inputs, self.size)

# Wrapper over the Sequential layer (analogous to CustomMPSSoftshrinkModel)
class CustomResizeModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(256), # Analogous to nn.Linear
            # Reshape to make it compatible with resize_bilinear (4D tensor required)
            tf.keras.layers.Reshape((16, 16, 1)),
            ResizeBilinearLayer(size=(32, 32)), # The target API
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10)
        ])

    def call(self, x):
        return self.model(x)

# Main execution block
if __name__ == "__main__":
    # Explicit device placement (analogous to mps_device = torch.device("mps"))
    with tf.device('/GPU:0'):
        # Create dummy input (batch of flattened vectors)
        input_data = tf.random.normal([2, 784])

        model = CustomResizeModel()
        output = model(input_data)

        # Verify execution
        assert output.shape == (2, 10)
        print("Test passed: resize_bilinear executed successfully on GPU.")