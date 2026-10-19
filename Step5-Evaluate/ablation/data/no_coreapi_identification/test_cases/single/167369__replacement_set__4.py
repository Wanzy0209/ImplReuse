import torch
import tensorflow as tf
import torch.nn.functional as F

# Replacement: torch.nn -> tf.keras.initializers.serialize
from tf.keras.initializers import serialize as nn

class Config(tf.keras.initializers.Initializer):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "Config()"

    def get_config(self):
        return {}

    def __call__(self, shape, dtype=None):
        return tf.ones(shape, dtype=dtype)

def forward(x, config):
    # Replacement: repr -> tf.keras.initializers.serialize
    return x * len(tf.keras.initializers.serialize(config))

config = Config()

# Replacement: torch.randn -> torch.nn.functional.adaptive_max_pool2d
# We need an input tensor for the pooling operation.
# Using torch.ones to create a dummy input.
dummy_input = torch.ones(1, 1, 4, 4)
x = torch.nn.functional.adaptive_max_pool2d(dummy_input, (2, 2))

compiled = torch.compile(forward, fullgraph=True)