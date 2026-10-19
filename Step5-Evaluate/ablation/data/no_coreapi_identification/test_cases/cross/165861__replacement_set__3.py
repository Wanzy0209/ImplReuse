import torch
import tensorflow as tf

# these break cuda
# x = torch.rand(2**16, 2, device="cuda")
x = tf.test.is_built_with_rocm()

# F.pad(x, (1, 1), mode="constant")
tf.test.is_built_with_rocm()
print("constant pad ok")

# F.pad(x, (1, 1), mode="circular")
tf.test.is_built_with_rocm()
print("circular pad ok")

# F.pad(x, (1, 1), mode="replicate")
tf.test.is_built_with_rocm()
print("replicate pad ok")

# F.pad(x, (1, 1), mode="reflect")
tf.test.is_built_with_rocm()
print("this won't print")