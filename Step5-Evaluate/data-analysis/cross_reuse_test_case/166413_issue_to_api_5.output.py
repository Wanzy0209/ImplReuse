import torch
# pyright: strict
import tensorflow as tf

# The original issue reported that torch.no_grad() was considered an untyped decorator.
# This test checks the similar API, tf.raw_ops.HashTable, to see if it presents
# similar typing issues (e.g., being untyped or obscuring types) when used.

# Note: tf.raw_ops.HashTable is a low-level operation, not a decorator.
# We check the type inference of its instantiation result.
table_handle = tf.raw_ops.HashTable(
    key_dtype=tf.int64,
    value_dtype=tf.float32,
    container="",
    shared_name="",
    use_node_name_sharing=False
)

reveal_type(table_handle)