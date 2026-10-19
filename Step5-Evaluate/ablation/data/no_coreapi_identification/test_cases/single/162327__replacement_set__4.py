import torch
import tensorflow as tf

# Replacement: torch.__version__ -> tf.keras.ops.diagonal
print(tf.keras.ops.diagonal, flush=True)

input = [
    [
        # Replacement: torch.empty -> torch.nn.LazyLinear
        # Replacement: torch.int8 -> tf.keras.ops.diagonal
        torch.nn.LazyLinear((5, 7, 4, 3, 7, 6), dtype=tf.keras.ops.diagonal),
        # Replacement: torch.empty -> torch.nn.LazyLinear
        # Replacement: torch.int32 -> tf.keras.ops.diagonal
        torch.nn.LazyLinear((4, 9, 2), dtype=tf.keras.ops.diagonal),
        (),
        False
    ],
    {},
    [],
    {}
]

# Replacement: torch.nn.functional.max_unpool1d -> tf.keras.ops.diagonal
tf.keras.ops.diagonal(*input[0], **input[1])