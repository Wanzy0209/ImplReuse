import torch
import tensorflow as tf
from typing import NamedTuple

class MyNamedTuple(NamedTuple):
    first: tf.Tensor
    second: tf.Tensor

class MyNamedTupleSubclass(MyNamedTuple):
    pass

def fn(tup: MyNamedTuple) -> MyNamedTuple:
    extra_info = tf.constant(4.0)
    tup.extra_info = extra_info  # Add dynamic attribute
    return tup

print("\nTesting NamedTuple without tf.name_scope:")
extended_tup = MyNamedTupleSubclass(first=tf.constant([2.0]), second=tf.constant(1.0))
setattr_result = fn(extended_tup)
print(f"NamedTuple result: {setattr_result.extra_info}")

print("\nTesting NamedTuple with tf.name_scope:")
extended_tup = MyNamedTupleSubclass(first=tf.constant([2.0]), second=tf.constant(1.0))
# Adaptation: Using tf.name_scope as the context wrapper, analogous to torch.compile in the original
with tf.name_scope("dynamic_attr_scope"):
    setattr_result = fn(extended_tup)
print(f"NamedTuple result: {setattr_result.extra_info}")