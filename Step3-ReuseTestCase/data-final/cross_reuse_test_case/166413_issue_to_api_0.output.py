import torch
# pyright: strict
import tensorflow as tf
from typing import reveal_type

# The original issue highlights that torch.no_grad is untyped, causing 
# pyright to report errors in strict mode when used as a decorator.
# The similar API, tf.keras.initializers.HeUniform, is a class.
# Based on the provided implementation snippet, its __init__ method 
# lacks type hints (seed=None). This test checks if HeUniform 
# suffers from similar typing deficiencies when instantiated 
# and inspected with reveal_type.

initializer = tf.keras.initializers.HeUniform(seed=42)
reveal_type(initializer)