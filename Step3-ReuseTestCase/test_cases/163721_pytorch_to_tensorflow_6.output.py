import torch
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from tensorflow.keras import layers, Model

# Check for GPU availability (mimicking torch.backends.mps.is_available())
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU available: {gpus[0].name}")
else:
    print("No GPU available, running on CPU.")

# Wrapper over the tf.experimental.numpy.isreal API.
class TFIsReal(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # Call the similar API
        return tnp.isreal(inputs)

# Wrapper over the Sequential layer, using the custom isreal implementation.
class CustomTFIsRealModel(Model):
    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ):
        super().__init__()

        # Note: isreal returns a boolean tensor. To maintain compatibility
        # with Dense layers (which expect float inputs), we cast the result back to float.
        # This preserves the structural logic of the original test case.
        self.model = tf.keras.Sequential([
            layers.Dense(lin1_size, input_shape=(input_size,)),
            TFIsReal(),
            layers.Lambda(lambda x: tf.cast(x, tf.float32)),
            layers.Dense(lin2_size),
            TFIsReal(),
            layers.Lambda(lambda x: tf.cast(x, tf.float32)),
            layers.Dense(lin3_size),
            TFIsReal(),
            layers.Lambda(lambda x: tf.cast(x, tf.float32)),
            layers.Dense(output_size),
        ])

    def call(self, x):
        return self.model(x)

if __name__ == "__main__":
    # Create dummy input with complex numbers to test isreal behavior
    # Imaginary part is zero, so isreal should return True
    x = tf.complex(tf.random.normal((32, 784)), tf.zeros((32, 784)))

    model = CustomTFIsRealModel()
    output = model(x)

    # Basic assertion to verify execution
    assert output.shape == (32, 10)
    print("Test case executed successfully.")