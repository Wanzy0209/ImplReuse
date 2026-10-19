```python
'''
Copyright © 2023 Apple Inc.

See LICENSE folder for this sample’s licensing information.

Abstract:
The code defining the custom and default soft shrink models.
'''

import tensorflow as tf
from tensorflow import keras
# from compiler import *  # Removed: PyTorch specific custom C++ binding

# Conversion: PyTorch checks for MPS (Apple Silicon GPU). 
# TensorFlow checks for general GPU availability (e.g., CUDA or Metal).
assert len(tf.config.list_physical_devices('GPU')) > 0
# Conversion: PyTorch uses explicit device objects. TensorFlow uses device strings or context managers.
device_name = '/GPU:0'  # Device string representing GPU.

# Wrapper over the soft shrink logic.
# Conversion: Replaced custom C++ kernel (compiled_lib.mps_softshrink) with TF ops implementation.
class MPSSoftshrink(keras.layers.Layer):
    # __constants__ is not needed in Keras
    lambd: float

    def __init__(self, lambd: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.lambd = lambd

    def call(self, inputs):
        # Softshrink: f(x) = x - lambda if x > lambda, x + lambda if x < -lambda, else 0
        return tf.where(
            inputs > self.lambd,
            inputs - self.lambd,
            tf.where(
                inputs < -self.lambd,
                inputs + self.lambd,
                tf.zeros_like(inputs)
            )
        )

    def get_config(self):
        config = super().get_config()
        config.update({"lambd": self.lambd})
        return config

    # extra_repr is implicitly handled by Keras layer printing, relying on get_config.

# Wrapper over the Sequential layer, using the custom soft shrink implementation.
class CustomMPSSoftshrinkModel(keras.Model):
    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ):
        super().__init__()

        # Conversion: nn.Sequential -> keras.Sequential
        # Conversion: nn.Linear -> keras.layers.Dense
        self.model = keras.Sequential([
            keras.layers.Dense(lin1_size, input_shape=(input_size,)),
            MPSSoftshrink(),
            keras.layers.Dense(lin2_size),
            MPSSoftshrink(),
            keras.layers.Dense(lin3_size),
            MPSSoftshrink(),
            keras.layers.Dense(output_size),
        ])

    def call(self, x):
        return self.model(x)
```