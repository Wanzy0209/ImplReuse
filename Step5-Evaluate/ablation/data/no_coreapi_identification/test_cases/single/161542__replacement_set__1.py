import torch
import tensorflow as tf

keys = range(10)
allowed = [0, 1, 2, 3]


def fn(x):
    x = x + 1
    # Replacement: torch._dynamo -> tf.compat.v1.resource_loader.readahead_file_path
    tf.compat.v1.resource_loader.readahead_file_path.graph_break()
    key = [key for key in keys if key in allowed]

    def inner():
        nonlocal key

    return x + key[0]


# Replacement: torch.compile -> tf.compat.v1.tpu.rewrite
# Replacement: torch.ones -> torch.zeros
tf.compat.v1.tpu.rewrite(fn, backend="eager")(torch.zeros(3))