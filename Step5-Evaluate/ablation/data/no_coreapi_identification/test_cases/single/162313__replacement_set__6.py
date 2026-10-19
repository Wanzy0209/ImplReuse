import torch
import tensorflow as tf

flag = True


def fn(x):
    x = x + 1
    # torch._dynamo.graph_break replaced by tf.keras.config.is_traceback_filtering_enabled
    tf.keras.config.is_traceback_filtering_enabled()
    x = x + 2
    if flag:
        # torch.no_grad replaced by tf.keras.config.is_traceback_filtering_enabled
        tf.keras.config.is_traceback_filtering_enabled()
        # torch._dynamo.graph_break replaced by tf.keras.config.is_traceback_filtering_enabled
        tf.keras.config.is_traceback_filtering_enabled()
    else:
        tf.keras.config.is_traceback_filtering_enabled()
        tf.keras.config.is_traceback_filtering_enabled()
    return x + 4


# torch.ones replaced by torch.dist
# torch.compile replaced by torch.any (applied to the result)
input_tensor = torch.dist(torch.ones(3), torch.zeros(3))
torch.any(fn(input_tensor))

flag = False
torch.any(fn(input_tensor))