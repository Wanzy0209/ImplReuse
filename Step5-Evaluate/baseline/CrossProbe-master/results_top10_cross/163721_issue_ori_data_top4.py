```python
'''
Copyright © 2023 Apple Inc.

See LICENSE folder for this sample’s licensing information.

Abstract:
The code defining the custom and default soft shrink models.
'''

import tensorflow as tf
# from compiler import *  # Conversion: Not needed in TF, custom logic implemented via TF ops below

# Conversion: Check for GPU availability (MPS on Mac maps to GPU in TensorFlow)
assert tf.config.list_physical_devices('GPU'), "GPU not available"
# Conversion: In TF, device placement is typically handled via context managers (with tf.device(...))
# rather than explicit device objects passed to tensors.
device_name = "/GPU:0"

# Wrapper over the custom MPS soft shrink kernel.
# Conversion: Inheriting from tf.keras.layers.Layer instead of nn.Module
class MPSSoftshrink(tf.keras.layers.Layer):
    # Conversion: __constants__ is not needed in TF
    lambd: float

    def __init__(self, lambd: float = 0.5) -> None:
        super(MPSSoftshrink, self).__init__()
        self.lambd = lambd

    # Conversion: forward() becomes call() in Keras layers
    def call(self, inputs):
        # Conversion: Implementing soft shrink logic using TF ops.
        # Original code called compiled_lib.mps_softshrink.
        # Logic: x - lambd if x > lambd, x + lambd if x < -lambd, else 0
        return tf.where(inputs > self.lambd, inputs - self.lambd,
                        tf.where(inputs < -self.lambd, inputs + self.lambd, tf.zeros_like(inputs)))

    # Conversion: extra_repr is replaced by get_config for serialization/representation
    def get_config(self):
        config = super(MPSSoftshrink, self).get_config()
        config.update({"lambd": self.lambd})
        return config

# Wrapper over the Sequential layer, using the custom MPS kernel soft shrink implementation.
# Conversion: Inheriting from tf.keras.Model instead of nn.Module
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

        # Conversion: nn.Sequential maps to tf.keras.Sequential
        # Conversion: nn.Linear maps to tf.keras.layers.Dense
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(lin1_size, input_shape=(input_size,)),
            MPSSoftshrink(),
            tf.keras.layers.Dense(lin2_size),
            MPSSoftshrink(),
            tf.keras.layers.Dense(lin3_size),
            MPSSoftshrink(),
            tf.keras.layers.Dense(output_size),
        ])

    # Conversion: forward() becomes call() in Keras models
    def call(self, x):
        return self.model(x)
```