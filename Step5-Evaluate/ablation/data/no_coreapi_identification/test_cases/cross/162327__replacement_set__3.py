import torch
import tensorflow as tf

# Replacement: torch.__version__ -> tf.experimental.numpy.diagonal
print(tf.experimental.numpy.diagonal, flush=True)

# Replacements:
# torch.empty -> torch.set_rng_state
# torch.int8 -> tf.experimental.numpy.diagonal
# torch.int32 -> tf.experimental.numpy.diagonal
input = [
    [
        torch.set_rng_state((5, 7, 4, 3, 7, 6), dtype=tf.experimental.numpy.diagonal),
        torch.set_rng_state((4, 9, 2), dtype=tf.experimental.numpy.diagonal),
        (),
        False
    ],
    {},
    [],
    {}
]

# Replacement: torch.nn.functional.max_unpool1d -> tf.experimental.numpy.diagonal
tf.experimental.numpy.diagonal(*input[0], **input[1])