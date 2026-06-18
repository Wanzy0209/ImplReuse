import torch
# pyright: strict
import tensorflow as tf

# The original issue highlights that untyped class methods (like __new__) cause
# type checkers to report errors or obscure types when the class is used as a decorator.
# The similar API, tf.keras.initializers.LecunNormal, shows an untyped __init__
# in its implementation (seed=None). This test verifies if the type checker can correctly
# infer the type of the initializer instance and its return value, mirroring the
# check performed on the torch.no_grad decorator.

# Test instantiation with an argument
initializer = tf.keras.initializers.LecunNormal(seed=42)
reveal_type(initializer)

# Test the callable behavior (Initializers are callable)
# This mirrors checking the decorated function's type in the original issue.
value = initializer(shape=(2, 2))
reveal_type(value)