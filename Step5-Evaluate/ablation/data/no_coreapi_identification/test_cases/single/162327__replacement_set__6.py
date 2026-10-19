import torch
import tensorflow as tf

# Replacement: torch.__version__ -> tf.saved_model.save
print(tf.saved_model.save, flush=True)

# Replacement: torch.empty -> torch.empty_strided
# Replacement: torch.int8 -> tf.saved_model.save
# Replacement: torch.int32 -> tf.saved_model.save
# Note: torch.empty_strided requires stride arguments. We calculate contiguous strides to maintain runnability.
# Strides for (5, 7, 4, 3, 7, 6) -> (3528, 504, 126, 42, 6, 1)
# Strides for (4, 9, 2) -> (18, 2, 1)
input = [
    [
        torch.empty_strided((5, 7, 4, 3, 7, 6), (3528, 504, 126, 42, 6, 1), dtype=tf.saved_model.save),
        torch.empty_strided((4, 9, 2), (18, 2, 1), dtype=tf.saved_model.save),
        (),
        False
    ],
    {},
    [],
    {}
]

# Replacement: torch.nn.functional.max_unpool1d -> tf.saved_model.save
tf.saved_model.save(*input[0], **input[1])