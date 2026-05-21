import torch
import tensorflow as tf

# Check for GPU availability (analogous to torch.backends.mps.is_available)
gpus = tf.config.list_physical_devices('GPU')
assert len(gpus) > 0, "GPU is not available"

# Wrapper over the custom soft shrink kernel (simulated with TF ops)
# In the original bug, this calls an external compiled library.
class TFSoftshrink(tf.keras.layers.Layer):
    def __init__(self, lambd: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.lambd = lambd

    def call(self, inputs):
        # Simulating the custom op logic
        return tf.where(inputs > self.lambd, inputs - self.lambd,
                        tf.where(inputs < -self.lambd, inputs + self.lambd, 0.0))

    def get_config(self):
        return {'lambd': self.lambd}

# Wrapper over the Sequential layer, using the custom kernel implementation.
class CustomTFSoftshrinkModel(tf.keras.Model):
    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ):
        super().__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(lin1_size, input_shape=(input_size,)),
            TFSoftshrink(),
            tf.keras.layers.Dense(lin2_size),
            TFSoftshrink(),
            tf.keras.layers.Dense(lin3_size),
            TFSoftshrink(),
            tf.keras.layers.Dense(output_size),
        ])

    def call(self, x):
        return self.model(x)

# Main execution logic
# Create a tensor on the GPU to serve as the anchor for colocation
with tf.device('/GPU:0'):
    dummy_input = tf.ones((1, 784))
    
    # Use tf.compat.v1.colocate_with to ensure the model construction and execution
    # happens on the same device as dummy_input.
    # This is the API under test, analogous to setting the device in PyTorch.
    with tf.compat.v1.colocate_with(dummy_input):
        model = CustomTFSoftshrinkModel()
        output = model(dummy_input)

# Verify execution completed successfully
assert output.shape == (1, 10)
print("Test passed.")