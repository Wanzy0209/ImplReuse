```python
'''
Copyright © 2023 Apple Inc.

See LICENSE folder for this sample’s licensing information.

Abstract:
The code defining the custom and default soft shrink models.
'''

import tensorflow as tf

# Conversion: torch.backends.mps.is_available() -> Check for GPU availability
# Note: TensorFlow checks for general GPU (CUDA/ROCm) availability.
assert len(tf.config.list_physical_devices('GPU')) > 0

# Conversion: torch.device("mps") -> Get device name
# Note: Using tensorflow.python.eager.context.get_device_name equivalent logic
mps_device = tf.test.gpu_device_name()

# Wrapper over the custom MPS soft shrink kernel.
# Conversion: nn.Module -> tf.keras.layers.Layer
class MPSSoftshrink(tf.keras.layers.Layer):
    # __constants__ is not needed in TensorFlow
    lambd: float

    def __init__(self, lambd: float = 0.5, **kwargs) -> None:
        super(MPSSoftshrink, self).__init__(**kwargs)
        self.lambd = lambd

    # Conversion: forward -> call
    def call(self, inputs):
        # Conversion: compiled_lib.mps_softshrink -> TF implementation of Softshrink
        # Softshrink(x) = x - lambd if x > lambd, x + lambd if x < -lambd, 0 otherwise
        return tf.where(
            inputs > self.lambd,
            inputs - self.lambd,
            tf.where(
                inputs < -self.lambd,
                inputs + self.lambd,
                0.0
            )
        )

    # Conversion: extra_repr -> TF layers handle this automatically via __str__ or get_config
    def get_config(self):
        config = super(MPSSoftshrink, self).get_config()
        config.update({"lambd": self.lambd})
        return config

# Wrapper over the Sequential layer, using the custom MPS kernel soft shrink implementation.
# Conversion: nn.Module -> tf.keras.Model
class CustomMPSSoftshrinkModel(tf.keras.Model):
    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ):
        super(CustomMPSSoftshrinkModel, self).__init__()

        # Conversion: nn.Sequential -> tf.keras.Sequential
        # Conversion: nn.Linear -> tf.keras.layers.Dense
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(lin1_size, input_shape=(input_size,)),
            MPSSoftshrink(),
            tf.keras.layers.Dense(lin2_size),
            MPSSoftshrink(),
            tf.keras.layers.Dense(lin3_size),
            MPSSoftshrink(),
            tf.keras.layers.Dense(output_size),
        ])

    # Conversion: forward -> call
    def call(self, x):
        return self.model(x)
```