import torch
import tensorflow as tf
from torch.nn import functional as F

# 1. Setup weight and bias
# Replacement: torch.rand -> tf.keras.activations.relu
# We generate random data in TensorFlow and apply ReLU to simulate the initialization,
# then convert to PyTorch MPS tensors.
w_tf = tf.random.uniform((768, 768), minval=-1.0, maxval=1.0)
w_tf = tf.keras.activations.relu(w_tf)
weight = torch.from_numpy(w_tf.numpy()).to("mps").float()

b_tf = tf.random.uniform((768,), minval=-1.0, maxval=1.0)
b_tf = tf.keras.activations.relu(b_tf)
bias = torch.from_numpy(b_tf.numpy()).to("mps").float()

# 2. Setup base tensor
# Replacement: torch.arange -> torch.svd
# We create a dummy matrix to perform SVD. 
# To facilitate the next step (bmm), we use a 3D matrix.
# torch.svd on a (5, 499, 768) tensor returns U (5, 499, 499), S (5, 499), V (5, 768, 768).
svd_input_matrix = torch.randn(5, 499, 768, device="mps")
U, S, V = torch.svd(svd_input_matrix)

# 3. Setup input tensor
# Replacement: torch.as_strided -> torch.bmm
# We use the outputs of the SVD operation to perform a batch matrix multiplication.
# bmm(U, V) results in a tensor of shape (5, 499, 768), matching the original input shape.
input = torch.bmm(U, V)

# 4. Call the API under test
# Original API: torch.nn.functional.linear
output = F.linear(input, weight, bias)

# Assertions to verify the test runs correctly
assert output is not None
assert output.shape == (5, 499, 768)
assert output.device.type == "mps"