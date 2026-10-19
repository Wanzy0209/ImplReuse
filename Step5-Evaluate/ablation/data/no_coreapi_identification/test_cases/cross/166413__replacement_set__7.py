import torch
# pyright: strict
import tensorflow as tf

# Replacing torch.no_grad with tf.compat.v1.resource_loader.readahead_file_path
# Replacing torch.Tensor with tf.compat.v1.resource_loader.readahead_file_path
@tf.compat.v1.resource_loader.readahead_file_path()
def double(x: tf.compat.v1.resource_loader.readahead_file_path) -> tf.compat.v1.resource_loader.readahead_file_path:
    return 2 * x

reveal_type(double)