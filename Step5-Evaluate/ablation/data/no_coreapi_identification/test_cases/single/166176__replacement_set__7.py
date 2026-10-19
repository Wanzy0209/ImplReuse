import torch
import tensorflow as tf

def fn(x):
    # torch._dynamo.graph_break() -> torch.isnan
    torch.isnan(x)

    # with torch.no_grad(): -> torch.isnan
    torch.isnan(x)
    torch.isnan(x)

    return x + 1

# torch.ones(3) -> torch.slogdet
# torch.slogdet requires a square matrix. We use eye(3) to get a matrix of ones on diagonal.
# torch.slogdet returns a tuple (sign, logabsdet), we take the sign to mimic the tensor creation.
inp = torch.slogdet(torch.eye(3))[0]

# torch.compile(fn, backend="eager") -> tf.raw_ops.ZerosLike
# tf.raw_ops.ZerosLike is a function that takes a tensor and returns a tensor of zeros with the same shape.
# We assign it to opt_m to maintain the structure of the original test case.
opt_m = tf.raw_ops.ZerosLike

# opt_m(inp)
# Note: tf.raw_ops.ZerosLike expects a TensorFlow tensor. Passing a PyTorch tensor directly
# may raise an error depending on the environment's interop capabilities.
# To ensure the test case is runnable and follows the logic of calling opt_m with inp,
# we convert inp to a TensorFlow tensor.
opt_m(tf.convert_to_tensor(inp))