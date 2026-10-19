```python
'''
Copyright © 2023 Apple Inc.

See LICENSE folder for this sample’s licensing information.

Abstract:
The code defining the custom and default soft shrink models.
'''

import tensorflow as tf
# from compiler import *  # Conversion: Removed external compiler dependency; using native TF ops.

# Conversion: Check for GPU availability (equivalent to torch.backends.mps.is_available)
gpus = tf.config.list_physical_devices('GPU')
assert len(gpus) > 0, "GPU is not available"
# Conversion: TensorFlow handles device placement implicitly or via context managers.
# We define the device string for potential explicit placement.
device_name = '/GPU:0'

# Wrapper over the custom MPS soft shrink kernel.
# Conversion: Inherit from tf.keras.layers.Layer instead of nn.Module.
class MPSSoftshrink(tf.keras.layers.Layer):
    # Conversion: __constants__ is not strictly required in TF, but we track the variable.
    lambd: float

    def __init__(self, lambd: float = 0.5, **kwargs) -> None:
        super(MPSSoftshrink, self).__init__(**kwargs)
        self.lambd = lambd

    # Conversion: forward() is replaced by call().
    def call(self, inputs):
        # Conversion: Replacing compiled_lib.mps_softshrink with native TensorFlow ops
        # to ensure the code is valid and runnable without external C++ compilation.
        # Softshrink formula: f(x) = x - lambd if x > lambd, x + lambd if x < -lambd, else 0
        return tf.where(
            inputs > self.lambd,
            inputs - self.lambd,
            tf.where(
                inputs < -self.lambd,
                inputs + self.lambd,
                tf.zeros_like(inputs)
            )
        )

    # Conversion: extra_repr is roughly equivalent to get_config for serialization in TF.
    def get_config(self):
        config = super(MPSSoftshrink, self).get_config()
        config.update({"lambd": self.lambd})
        return config

    def __str__(self):
        return f"MPSSoftshrink({self.lambd})"

# Wrapper over the Sequential layer, using the custom MPS kernel soft shrink implementation.
# Conversion: Inherit from tf.keras.Model.
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

        # Conversion: nn.Sequential is replaced by tf.keras.Sequential.
        # Conversion: nn.Linear is replaced by tf.keras.layers.Dense.
        # Note: Dense units correspond to output features. Input shape is specified in the first layer
        # to allow model building without needing a dummy input call first.
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(lin1_size, input_shape=(input_size,)),
            MPSSoftshrink(),
            tf.keras.layers.Dense(lin2_size),
            MPSSoftshrink(),
            tf.keras.layers.Dense(lin3_size),
            MPSSoftshrink(),
            tf.keras.layers.Dense(output_size),
        ])

    # Conversion: forward() is replaced by call().
    def call(self, x):
        return self.model(x)
```