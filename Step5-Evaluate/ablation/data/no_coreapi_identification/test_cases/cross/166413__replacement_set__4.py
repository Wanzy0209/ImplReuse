# pyright: strict
import tensorflow as tf

# Note: tf.keras.layers.Wrapper is an abstract class and cannot be instantiated directly.
# We wrap the definition in a try-except block to ensure the script remains runnable
# while preserving the logic for static type checking (pyright).
try:
    @tf.keras.layers.Wrapper
    def double(x: tf.keras.layers.Wrapper) -> tf.keras.layers.Wrapper:
        return x
except TypeError:
    # Expected runtime error: Can't instantiate abstract class Wrapper
    pass

reveal_type(double)