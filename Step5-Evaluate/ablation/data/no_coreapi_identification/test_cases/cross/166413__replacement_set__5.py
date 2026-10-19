# pyright: strict
import tensorflow as tf

@tf.keras.initializers.LecunNormal()
def double(x: tf.keras.initializers.LecunNormal) -> tf.keras.initializers.LecunNormal:
    return 2 * x

reveal_type(double)