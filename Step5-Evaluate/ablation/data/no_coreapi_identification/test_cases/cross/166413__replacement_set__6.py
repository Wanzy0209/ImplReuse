# pyright: strict
import tensorflow as tf

@tf.raw_ops.HashTable(key_dtype=tf.int32, value_dtype=tf.int32)
def double(x: tf.raw_ops.HashTable) -> tf.raw_ops.HashTable:
    return 2 * x

reveal_type(double)