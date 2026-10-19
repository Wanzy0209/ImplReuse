import torch
import torchvision
import tensorflow as tf

model = torchvision.models.mobilenet_v2(weights=None)

# Replacement: torch.rand -> tf.keras.ops.relu
# To maintain the logic of creating an input tensor 'x', we generate a random tensor
# using TensorFlow, apply the replacement API (relu), and convert it to a PyTorch tensor.
x = torch.from_numpy(tf.keras.ops.relu(tf.random.uniform((1, 3, 224, 224))).numpy())

ep = torch.export.export(model, (x,))

# Replacement: torch.testing.assert_close -> torch.nn.functional.cross_entropy
# Instead of asserting closeness, we calculate the cross entropy loss between the
# outputs of the original model and the exported model.
loss = torch.nn.functional.cross_entropy(model(x), ep.module()(x))
print(f"Cross entropy loss: {loss.item()}")