import torch
import torchvision
import tensorflow as tf
import torch.nn.utils.rnn

# Setup model
model = torchvision.models.mobilenet_v2(weights=None)

# Replacement: torch.rand -> tf.keras.backend.relu
# To maintain the logic of generating an input tensor 'x' of shape (1, 3, 224, 224),
# we create a tensor and apply the replacement function.
# We convert to numpy for TF interop and back to torch for the model.
# Using torch.randn to provide values to relu.
x = torch.from_numpy(tf.keras.backend.relu(torch.randn((1, 3, 224, 224)).numpy()).numpy())

# The API under test
ep = torch.export.export(model, (x,))

# Replacement: torch.testing.assert_close -> torch.nn.utils.rnn.pack_sequence
# We replace the assertion call with the pack_sequence call.
# The arguments are adapted to fit pack_sequence (list of tensors).
torch.nn.utils.rnn.pack_sequence([model(x), ep.module()(x)])