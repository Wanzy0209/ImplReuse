```python
'''
Copyright © 2023 Apple Inc.

See LICENSE folder for this sample’s licensing information.

Abstract:
The code defining the custom and default soft shrink models.
'''

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

# Conversion: Check for GPU availability. On macOS, TensorFlow utilizes the GPU device
# which maps to the Metal Performance Shaders (MPS) backend if tensorflow-metal is installed.
assert len(tf.config.list_physical_devices('GPU')) > 0
# Conversion: TF uses device strings (e.g., '/GPU:0') rather than device objects for most logic.
device_name = '/GPU:0'

# Wrapper over the custom MPS soft shrink kernel.
# Conversion: Inherits from tf.keras.layers.Layer instead of nn.Module.
class MPSSoftshrink(layers.Layer):
    # Conversion: Keras handles serialization via get_config, __constants__ is not needed.
    lambd: float

    def __init__(self, lambd: float = 0.5, **kwargs) -> None:
        super(MPSSoftshrink, self).__init__(**kwargs)
        self.lambd = lambd

    # Conversion: forward() becomes call()
    def call(self, inputs):
        # Original code calls: compiled_lib.mps_softshrink(input, self.lambd)
        # Since the custom C++ library is not available, we implement the Softshrink logic
        # using standard TensorFlow operations to ensure valid code.
        # f(x) = x - lambda if x > lambda, x + lambda if x < -lambda, else 0.
        return tf.where(inputs > self.lambd, inputs - self.lambd,
                        tf.where(inputs < -self.lambd, inputs + self.lambd, tf.zeros_like(inputs)))

    # Conversion: extra_repr is replaced by get_config for serialization and layer display.
    def get_config(self):
        config = super(MPSSoftshrink, self).get_config()
        config.update({"lambd": self.lambd})
        return config

# Wrapper over the Sequential layer, using the custom MPS kernel soft shrink implementation.
# Conversion: Inherits from tf.keras.Model.
class CustomMPSSoftshrinkModel(Model):
    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ):
        super(CustomMPSSoftshrinkModel, self).__init__()

        # Conversion: nn.Sequential becomes tf.keras.Sequential.
        # Conversion: nn.Linear becomes tf.keras.layers.Dense.
        self.model = Sequential([
            layers.Dense(lin1_size, input_shape=(input_size,)),
            MPSSoftshrink(),
            layers.Dense(lin2_size),
            MPSSoftshrink(),
            layers.Dense(lin3_size),
            MPSSoftshrink(),
            layers.Dense(output_size),
        ])

    # Conversion: forward() becomes call()
    def call(self, x):
        return self.model(x)
```