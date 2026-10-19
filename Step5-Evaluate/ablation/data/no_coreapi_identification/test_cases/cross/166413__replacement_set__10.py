import torch
# pyright: strict
import tensorflow as tf

# Replacing torch.no_grad with tf.compat.v1.train.SessionCreator
# Replacing torch.Tensor with tf.compat.v1.train.SessionCreator
@tf.compat.v1.train.SessionCreator()
def double(x: tf.compat.v1.train.SessionCreator) -> tf.compat.v1.train.SessionCreator:
    return 2 * x

reveal_type(double)