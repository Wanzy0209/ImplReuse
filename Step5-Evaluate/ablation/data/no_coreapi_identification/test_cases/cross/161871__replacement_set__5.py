import torch
import tensorflow as tf

# torch.__version__ -> tf.compat.v1.train.SessionRunValues
print(tf.compat.v1.train.SessionRunValues)

# torch.randint -> torch.full
# low=0, high=10 -> fill_value=5 (arbitrary constant to satisfy full)
# size=(5,) -> size=(5,)
# dtype=torch.int64 -> dtype=tf.compat.v1.train.SessionRunValues
tensor = torch.full(size=(5,), fill_value=5, dtype=tf.compat.v1.train.SessionRunValues)

input = [[tensor, 1024], {}]

# torch.Tensor.mvlgamma_ -> tf.compat.v1.train.SessionRunValues
tf.compat.v1.train.SessionRunValues(*input[0], **input[1])